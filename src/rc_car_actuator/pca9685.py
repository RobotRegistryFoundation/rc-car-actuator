"""Drive an ESC and a steering servo through a PCA9685 over I2C.

WHY A SECOND HARDWARE BACKEND. `PWMDrive` in `drive.py` speaks to pigpio, which
generates PWM from the Pi's own GPIO. That works, and it is not what is on the
bench: the hardware here is a PCA9685 breakout on I2C, which is the usual way an
RC-car kit is wired because it drives sixteen channels from two wires and keeps
its timing in its own oscillator rather than in a daemon on a busy Linux box.

THE PROPERTY THAT MAKES THAT LAST PART BOTH GOOD AND DANGEROUS. The PCA9685
holds its output in hardware. Once a pulse width is written the chip emits it
forever — through a Python exception, through the actuator process being killed,
through the Pi rebooting, through the kernel panicking. Nothing about a stopped
program stops the car. Every design decision below follows from that:

  * Neutral is written in `__init__`, before the object exists, so no code path
    can command motion through a channel that was never centred.
  * `neutral()` writes throttle FIRST, because if the second write fails the car
    is at least already stopped.
  * `off()` exists and is distinct from neutral: it stops the PWM signal
    altogether. Some ESCs treat signal-loss as a stop, and some hold their last
    command — which is exactly why the runbook requires that behaviour be tested
    on a stand rather than assumed.
  * NONE OF THIS IS THE SAFETY SYSTEM. The mushroom switch on the battery lead
    is the safety system. This file cannot survive the failures it is most
    needed for.

UNVERIFIED AGAINST HARDWARE at the time of writing: written from the PCA9685
datasheet against a fake bus. The pulse-width numbers are the hobby-RC standard
and MUST be checked against the actual ESC before the wheels touch the ground.
"""
from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from typing import Protocol

from .drive import clamp

logger = logging.getLogger("rc_car.pca9685")

# Registers, from the PCA9685 datasheet.
_MODE1 = 0x00
_MODE2 = 0x01
_PRESCALE = 0xFE
_LED0_ON_L = 0x06
_ALL_LED_ON_L = 0xFA

_MODE1_RESTART = 0x80
_MODE1_AI = 0x20        # auto-increment, so a channel writes as one 4-byte run
_MODE1_SLEEP = 0x10
_MODE2_OUTDRV = 0x04    # totem-pole outputs, which is what a servo header wants

#: The chip's nominal internal oscillator. See `Channel.oscillator_hz` for why
#: this is a starting point and not a constant.
NOMINAL_OSCILLATOR_HZ = 25_000_000

#: Servo/ESC frame rate. 50 Hz is the hobby-RC standard; a 20 ms frame with a
#: 1000-2000 us pulse inside it.
DEFAULT_FRAME_HZ = 50


class I2CBus(Protocol):
    """The two smbus2 calls this needs, and nothing else.

    Narrow on purpose: a protocol this small is trivial to fake, which is what
    lets every rule in this file be tested on a machine with no I2C bus at all.
    """

    def write_byte_data(self, addr: int, register: int, value: int) -> None: ...
    def read_byte_data(self, addr: int, register: int) -> int: ...


@dataclass(frozen=True)
class Channel:
    """One PWM output, and the numbers that make it mean something.

    A channel is not just a number, because two identical-looking servo headers
    on the same board disagree about what "centre" is. The ESC's neutral is
    whatever it was taught during its calibration; the steering servo's centre is
    wherever the linkage happens to put the wheels straight. Both are per-vehicle
    measurements, and hardcoding 1500 us for both is how a car creeps forward at
    rest and tracks 5 degrees left.
    """

    index: int
    #: Pulse width that means stop / straight ahead, microseconds.
    neutral_us: float = 1500.0
    #: Microseconds added at full command. The hobby standard is 500 (so
    #: 1000-2000), but a steering linkage frequently binds before full travel,
    #: and a servo pushing against a bind stalls, heats, and dies.
    span_us: float = 500.0
    #: Flip the direction of this channel. Which way "forward" is depends on how
    #: the ESC is wired and which way round the servo horn went on.
    invert: bool = False

    def pulse_us(self, value: float) -> float:
        """Pulse width for a command in -1..1, clamped and NaN-safe."""
        v = clamp(value)
        if self.invert:
            v = -v
        return self.neutral_us + v * self.span_us


@dataclass(frozen=True)
class DriveChannels:
    throttle: Channel
    steering: Channel
    frame_hz: int = DEFAULT_FRAME_HZ
    #: The chip's ACTUAL oscillator frequency.
    #:
    #: 🔴 THE CALIBRATION THAT WILL BITE FIRST. The PCA9685's internal oscillator
    #: is specified as 25 MHz but is a cheap on-die RC oscillator with several
    #: percent of tolerance, and every pulse width the chip emits is scaled by
    #: it. A part running 4% fast turns a requested 1500 us neutral into roughly
    #: 1440 us, which many ESCs read as a slow crawl in reverse — a car that
    #: moves while commanded to stop, from a bug that is invisible in the code.
    #:
    #: The measurement is simple and worth doing once per board: command a known
    #: pulse, measure the real frame period on a scope or a logic analyser, and
    #: set this to `nominal * (measured_period / requested_period)`. Until then
    #: the nominal value is used and the CAR STAYS ON A STAND.
    oscillator_hz: int = NOMINAL_OSCILLATOR_HZ


class PCA9685Drive:
    """`DriveHardware` over a PCA9685. Same contract as `SimulatedDrive`.

    Constructing this ARMS THE HARDWARE: it resets the chip, sets the frame rate,
    and writes neutral to both channels. An ESC that wants neutral held at
    power-on to arm gets it for free, because neutral is the first thing written
    and nothing changes it until a command arrives.
    """

    def __init__(self, bus: I2CBus, address: int = 0x40,
                 channels: DriveChannels | None = None) -> None:
        self._bus = bus
        self._address = address
        self._ch = channels or DriveChannels(throttle=Channel(0), steering=Channel(1))
        self._lock = threading.Lock()

        if self._ch.throttle.index == self._ch.steering.index:
            # Worth refusing rather than tolerating: one channel driving both an
            # ESC and a servo means every steering input is also a throttle
            # input, and the first left turn is a launch.
            raise ValueError("throttle and steering cannot share a PCA9685 channel")

        self._configure()
        # Before the object exists, so nothing can drive through an uncentred
        # channel. If this raises, there is no PCA9685Drive to command.
        self.neutral()

    # -- setup ---------------------------------------------------------------

    def _configure(self) -> None:
        """Reset, set the frame rate, wake up.

        The sleep/wake dance is required, not defensive: the prescale register is
        write-protected unless the chip is asleep, so a frequency written to an
        awake chip is silently ignored and the outputs keep running at whatever
        the last frame rate was — 200 Hz out of a reset, which a servo reads as
        a permanent full-travel command.
        """
        with self._lock:
            self._write(_MODE1, _MODE1_SLEEP)
            self._write(_MODE2, _MODE2_OUTDRV)
            self._write(_PRESCALE, self.prescale(self._ch.frame_hz,
                                                 self._ch.oscillator_hz))
            self._write(_MODE1, _MODE1_AI)
            # The datasheet's 500 us oscillator settling time after clearing
            # SLEEP. Reading a register back costs about that long over I2C and
            # is honest about waiting, where a bare sleep() in a driver tends to
            # get "optimised" away by somebody who does not know why it is there.
            self._read(_MODE1)
            self._write(_MODE1, _MODE1_AI | _MODE1_RESTART)

    @staticmethod
    def prescale(frame_hz: int, oscillator_hz: int = NOMINAL_OSCILLATOR_HZ) -> int:
        """Datasheet prescale for a frame rate, clamped to the legal 3..255.

        Clamped rather than allowed to wrap: an out-of-range value written to
        this register does not produce an error, it produces a frame rate nobody
        asked for, on hardware that is about to hold that value forever.
        """
        if frame_hz <= 0:
            raise ValueError("frame rate must be positive")
        value = round(oscillator_hz / (4096.0 * frame_hz)) - 1
        return int(max(3, min(255, value)))

    def counts(self, pulse_us: float) -> int:
        """Turn a pulse width into the chip's 12-bit off-count.

        Clamped to 0..4095 INSIDE the frame. A pulse longer than the frame is not
        a very long pulse — the counter wraps and it becomes a short one, which
        would turn "full forward" into something near neutral or worse.
        """
        # Checked BEFORE the arithmetic, not after: `round(nan)` raises rather
        # than returning nan, so a check on the result never runs. A NaN can
        # reach here from a trim value read out of a config file, which is
        # exactly the path that will not be exercised on the bench.
        if not math.isfinite(pulse_us):
            return 0
        period_us = 1_000_000.0 / self._ch.frame_hz
        counts = round(4096.0 * pulse_us / period_us)
        return int(max(0, min(4095, counts)))

    # -- the DriveHardware contract -----------------------------------------

    def set_drive(self, throttle: float, steering: float) -> None:
        with self._lock:
            self._write_channel(self._ch.steering, steering)
            # Throttle LAST on the way up, so a failure partway through leaves
            # the car not-moving rather than moving-and-unsteerable.
            self._write_channel(self._ch.throttle, throttle)

    def neutral(self) -> None:
        with self._lock:
            try:
                self._write_channel(self._ch.throttle, 0.0)
            finally:
                # Runs even if stopping raised, because a failure to stop is
                # exactly when the second write matters least and the first
                # matters most — and swallowing the steering write here would
                # leave the wheels locked over for whatever happens next.
                self._write_channel(self._ch.steering, 0.0)

    def off(self) -> None:
        """Stop emitting PWM on every channel.

        NOT a synonym for neutral, and choosing between them is a decision about
        the specific ESC on the bench: some read signal-loss as a stop, and some
        hold their last commanded throttle indefinitely. Verify which one is
        wired up, on a stand, before relying on either.
        """
        with self._lock:
            # The full-off bit in ALL_LED_OFF_H, which the datasheet says takes
            # precedence over any per-channel value.
            self._write(_ALL_LED_ON_L + 3, 0x10)

    # -- wire ----------------------------------------------------------------

    def _write_channel(self, channel: Channel, value: float) -> None:
        off = self.counts(channel.pulse_us(value))
        base = _LED0_ON_L + 4 * channel.index
        # ON is always 0: the pulse starts at the top of every frame and its
        # width is the OFF count. Phase-shifting channels would spread the
        # current draw of sixteen servos, and would also mean a partially-written
        # channel emits a pulse of the wrong length rather than the right one.
        self._write(base, 0x00)
        self._write(base + 1, 0x00)
        self._write(base + 2, off & 0xFF)
        self._write(base + 3, (off >> 8) & 0x0F)

    def _write(self, register: int, value: int) -> None:
        self._bus.write_byte_data(self._address, register, int(value) & 0xFF)

    def _read(self, register: int) -> int:
        return self._bus.read_byte_data(self._address, register)


def open_smbus_drive(address: int = 0x40, i2c_bus: int = 1,
                     channels: DriveChannels | None = None) -> PCA9685Drive:
    """Construct a `PCA9685Drive` on a real I2C bus.

    Kept out of `PCA9685Drive.__init__` so the class itself never imports
    smbus2. That is what lets every test above run on a laptop, on the Pi
    without the hardware attached, and in CI — the same code that will drive the
    car, exercised everywhere.
    """
    import smbus2  # imported lazily: hardware-only dependency

    return PCA9685Drive(smbus2.SMBus(i2c_bus), address=address, channels=channels)
