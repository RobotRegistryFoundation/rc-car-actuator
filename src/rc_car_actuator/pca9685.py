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
import time
from dataclasses import dataclass
from typing import Protocol

from .drive import Channel, clamp

__all__ = ["Channel", "DriveChannels", "PCA9685Drive", "I2CBus",
           "NOMINAL_OSCILLATOR_HZ", "DEFAULT_FRAME_HZ", "open_smbus_drive"]

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

#: Absolute pulse-width bounds, microseconds. Below/above these risks damaging a
#: servo or confusing an ESC into a state it will not leave.
#:
#: Ported from castor's own PCA9685 RC driver rather than reinvented — the same
#: numbers, for the same reason. They are a HARD clamp at the write, not a
#: validation at config time, because the trim values are per-vehicle and can be
#: edited by hand: a fat-fingered `neutral_us=15000` must become a safe pulse,
#: not a 15 ms one.
PULSE_MIN_US = 500
PULSE_MAX_US = 2500


def clamp_pulse_us(pulse_us: float) -> float:
    """Constrain a pulse width to what a servo or ESC can survive."""
    if pulse_us != pulse_us:      # NaN
        return float(PULSE_MIN_US)
    return max(float(PULSE_MIN_US), min(float(PULSE_MAX_US), float(pulse_us)))


class I2CBus(Protocol):
    """The two smbus2 calls this needs, and nothing else.

    Narrow on purpose: a protocol this small is trivial to fake, which is what
    lets every rule in this file be tested on a machine with no I2C bus at all.
    """

    def write_byte_data(self, addr: int, register: int, value: int) -> None: ...
    def read_byte_data(self, addr: int, register: int) -> int: ...


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

    #: Seconds of held neutral at construction, so an ESC recognises the signal
    #: and arms. Ported from castor's RC driver, which waits 0.5 s. Costs half a
    #: second once, at startup, OUTSIDE the command path — an ESC that never
    #: armed simply ignores everything afterwards, which reads as dead wiring.
    arm_delay_s: float = 0.5

    #: Most hobby ESCs ignore a reverse pulse unless they get a brief neutral
    #: first, and some need two neutral→reverse cycles.
    #:
    #: 🔴 OFF BY DEFAULT HERE, unlike castor's driver, and the difference is the
    #: layer. castor's runs in a request handler; this one sits under a deadman
    #: with a 50 ms watchdog, and the arming sequence SLEEPS. A stop that had to
    #: wait for an ESC handshake would be a stop that arrives up to 600 ms late,
    #: and "stopping never waits" is the one rule this package is built around.
    #: Turn it on once you know your ESC needs it; the sequence is abortable and
    #: a stop cuts it short (see `set_drive`).
    esc_reverse_arming: bool = False
    esc_arm_neutral_ms: int = 200
    esc_double_tap_reverse: bool = False
    #: Below this magnitude a throttle request counts as neutral, so noise around
    #: zero does not repeatedly re-trigger the reverse handshake.
    throttle_deadzone: float = 0.02


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

        # Set BEFORE neutral(), which consults it.
        self._stop_requested = threading.Event()
        self._last_throttle = 0.0

        # STARTS EVEN IF THE CHIP IS NOT THERE, and says so.
        #
        # On this vehicle the PCA9685's supply comes off the ESC's BEC, so the
        # chip is ABSENT FROM THE BUS whenever the RC pack is flat, unplugged or
        # on charge — which is most of the time a person is working on the car.
        # Construction used to raise there, and the gateway died on startup and
        # crash-looped: no status, no telemetry, no way to ask the robot what was
        # wrong. A robot that vanishes when a battery goes flat is far harder to
        # diagnose than one that answers "0x40 is not responding".
        #
        # Refusing to construct also bought nothing. Nothing can be driven
        # through an absent chip in any case — `set_drive` raises on the same
        # bus error — so the only thing the old behaviour prevented was FINDING
        # OUT. Actuation still fails closed; only observability changed.
        self._ready = False
        self._bring_up()

    def _bring_up(self) -> bool:
        """Configure, centre and arm. True if the chip answered.

        Retried on demand rather than once at startup, so plugging the pack back
        in recovers the drive without restarting the service.
        """
        try:
            self._configure()
            # Before anything can be commanded, so nothing drives through an
            # uncentred channel.
            self.neutral()
        except OSError as exc:
            self._ready = False
            logger.error("PCA9685 at 0x%02x is not answering (%s) — the drive "
                         "layer is present but unusable until it returns; on "
                         "this vehicle that usually means the RC pack is off",
                         self._address, exc)
            return False
        # Hold that neutral long enough for an ESC to arm. Only when the chip
        # has just come up, so it never delays a command or a stop.
        if not self._ready and self._ch.arm_delay_s > 0:
            time.sleep(self._ch.arm_delay_s)
            logger.info("ESC arming: held neutral for %.2fs", self._ch.arm_delay_s)
        self._ready = True
        return True

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
        # NEVER COMMAND A CHIP THAT WAS NEVER CONFIGURED.
        #
        # Present on the bus is not the same as ready. Out of reset the PCA9685
        # runs at 200 Hz, which a servo reads as a permanent full-travel
        # command, and writing a perfectly correct pulse WIDTH into a frame that
        # short produces full lock rather than the angle asked for. Letting
        # construction survive an absent chip opened exactly this hole — the
        # chip can now come back mid-session, answering writes while still
        # unconfigured — and this package's own test caught it.
        if not self._ready and not self._bring_up():
            raise OSError(
                f"PCA9685 at 0x{self._address:02x} is not configured and did "
                f"not answer; refusing to command an unconfigured chip")
        # A new drive command means driving is intended again, so any stop that
        # aborted a previous arming sequence stops suppressing this one.
        self._stop_requested.clear()
        if self._needs_reverse_arming(throttle):
            if self._arm_reverse(throttle):
                # A STOP ARRIVED DURING THE HANDSHAKE, SO DO NOT DRIVE.
                #
                # Caught by this package's own test. Abandoning the handshake
                # was not enough: control fell through to the throttle write
                # below and the car pulled away AFTER a stop — which is the one
                # outcome nothing in this package is allowed to produce. The
                # wheels are already neutral; the stop put them there.
                self._last_throttle = 0.0
                return
        with self._lock:
            self._write_channel(self._ch.steering, steering)
            # Throttle LAST on the way up, so a failure partway through leaves
            # the car not-moving rather than moving-and-unsteerable.
            self._write_channel(self._ch.throttle, throttle)
        self._last_throttle = clamp(throttle)

    def _needs_reverse_arming(self, throttle: float) -> bool:
        """Only on the forward→reverse transition, and only if configured."""
        if not self._ch.esc_reverse_arming:
            return False
        dead = self._ch.throttle_deadzone
        return clamp(throttle) < -dead and self._last_throttle >= -dead

    def _arm_reverse(self, throttle: float) -> bool:
        """Neutral, pause, (optionally reverse-neutral again), then let the caller drive.

        Returns True when a stop cut the sequence short, in which case the
        caller must NOT go on to drive.

        THE SLEEPS DO NOT HOLD THE LOCK, and every pause is really a wait on the
        stop event. That is what keeps this compatible with a deadman: a stop
        arriving mid-handshake takes the lock immediately, writes neutral, and
        cuts the remaining steps rather than queueing behind them.
        """
        step = max(0.0, self._ch.esc_arm_neutral_ms / 1000.0)

        def pause() -> bool:
            """True when a stop arrived and the sequence should be abandoned."""
            return self._stop_requested.wait(timeout=step)

        with self._lock:
            self._write_channel(self._ch.throttle, 0.0)
        if pause():
            return True
        if not self._ch.esc_double_tap_reverse:
            return False
        for value in (throttle, 0.0):
            if self._stop_requested.is_set():
                return True
            with self._lock:
                self._write_channel(self._ch.throttle, value)
            if pause():
                return True
        return False

    def neutral(self) -> None:
        # Set FIRST, so an arming sequence sleeping between writes abandons the
        # rest of itself instead of driving again after this returns.
        self._stop_requested.set()
        self._last_throttle = 0.0
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
        # Clamped to the absolute survivable range before anything reaches the
        # chip, so a mistyped trim in a config file cannot emit a pulse that
        # damages a servo.
        off = self.counts(clamp_pulse_us(channel.pulse_us(value)))
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

    def reachable(self) -> str | None:
        """None if the chip answers, else why it does not.

        MEASURED, NOT ASSUMED. On the bench the PCA9685 dropped off the bus
        mid-session — `i2cdetect` lost 0x40 while the fuel gauge two addresses
        down kept answering, so the bus was fine and the chip was not. Every
        `drive.set` then failed with `OSError 121 Remote I/O error`, and
        `status.report` went on saying `hardware: PCA9685Drive`, throttle 0.0,
        no error field: a perfectly healthy-looking robot that could not move a
        wheel. The phone would have shown green.

        A one-byte MODE1 read, so it is cheap enough to run on every telemetry
        sample. Reporting reachability is not the same as reporting that the
        WHEELS work — the ESC, its battery and the motor are all past this point
        and none of them answer questions.
        """
        try:
            self._read(_MODE1)
        except Exception as exc:  # noqa: BLE001 - any bus error means absent
            self._ready = False
            return f"{type(exc).__name__}: {exc}"
        # The chip is answering but was never configured — it came back after
        # being away, so bring it up rather than reporting a healthy chip that
        # is still running at whatever frame rate its reset left it at. 200 Hz
        # out of a reset is a permanent full-travel command to a servo.
        if not self._ready and not self._bring_up():
            return "present but could not be configured"
        return None


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
