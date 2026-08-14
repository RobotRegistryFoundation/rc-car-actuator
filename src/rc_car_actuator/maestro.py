"""Drive an ESC and a steering servo through a Pololu Maestro over USB serial.

WHY A THIRD BACKEND, WHEN THE PCA9685 ALREADY WORKS. Not for wiring convenience,
and not for power — a PCA9685's logic side draws about 10 mA and cannot brown
anything out. The reason is a property the PCA9685 does not have at any price:

    THE MAESTRO CAN BE CONFIGURED TO STOP ON ITS OWN.

A PCA9685 holds its last pulse width in hardware forever — through an exception,
a `kill -9`, a kernel panic, a reboot. Every stop in this system is therefore
software: the deadman thread writes neutral, and if the thing hosting that
thread dies, nothing does. The Maestro has a **serial timeout** setting and a
per-channel **"on startup or error"** position. Set them, and the controller
itself returns the wheels to neutral when commands stop arriving, with no Linux
in the path at all. That is a second, independent layer between the software
deadman and the mushroom switch, and it is the only reason this file exists.

🔴 THE LIMITATION THAT MATTERS, STATED HERE SO NOBODY INFERS OTHERWISE: that
timeout is a setting stored ON THE DEVICE, written with Pololu's Maestro Control
Center, and it is **not readable over the serial protocol**. This code cannot
confirm it is switched on, cannot switch it on, and must never be described as
providing it. Selecting this backend buys the failsafe only if a person
configured the device. Verify it the way the runbook verifies everything else —
by pulling the USB cable on a stand and watching the wheels — not by asking the
software.

PROTOCOL NOTES, all of them things that are easy to get wrong once:

  * TARGETS ARE IN QUARTER-MICROSECONDS. 1500 us is 6000, not 1500. A value sent
    in microseconds lands at a quarter of the intended pulse, which is below
    every servo's minimum and reads as full reverse lock.
  * EVERY DATA BYTE MUST HAVE BIT 7 CLEAR. The high bit distinguishes command
    bytes from data, so a target that leaks into it does not become a wrong
    position — it becomes a different COMMAND.
  * THE MAESTRO PRESENTS TWO USB SERIAL PORTS. The lower-numbered one
    (`/dev/ttyACM0`) is the Command Port and is the one to open; the other is
    the TTL port. Opening the wrong one looks exactly like a dead device.
  * The device must be in "USB Dual Port" serial mode. In "USB Chained" mode
    bytes written to the command port are forwarded out of the TTL pin instead,
    and again nothing moves and nothing errors.

UNVERIFIED AGAINST HARDWARE: written from Pololu's protocol documentation
against a fake port.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Protocol

from .drive import Channel

logger = logging.getLogger("rc_car.maestro")

# Compact-protocol command bytes.
_SET_TARGET = 0x84
_SET_SPEED = 0x87
_SET_ACCELERATION = 0x89
_GET_POSITION = 0x90
_GET_ERRORS = 0xA1
_GO_HOME = 0xA2

#: Target units per microsecond. The Maestro counts quarter-microseconds.
QUARTER_US = 4

#: Target 0 is special: it means "stop sending pulses on this channel" rather
#: than "drive to position zero". It is how `off()` is expressed.
TARGET_OFF = 0


class SerialPort(Protocol):
    """The two pyserial calls this needs, and nothing else.

    As narrow as the I2C protocol next door, and for the same reason: a
    two-method interface is trivial to fake, so every rule below is tested on a
    machine with no Maestro plugged into it.
    """

    def write(self, data: bytes) -> int: ...
    def read(self, size: int) -> bytes: ...


@dataclass(frozen=True)
class MaestroChannels:
    throttle: Channel
    steering: Channel
    #: Set when several Maestros share one TTL line, which switches the wire
    #: format to the Pololu protocol. Left None for the ordinary case of one
    #: controller on its own USB port.
    device_number: int | None = None
    #: Optional per-channel speed limit, in (quarter-us)/(10 ms). Zero is
    #: unlimited, which is correct for a THROTTLE — rate-limiting the throttle
    #: here would silently fight the deadman, since a neutral command that ramps
    #: is a neutral command that has not arrived yet.
    steering_speed: int = 0


class MaestroDrive:
    """`DriveHardware` over a Pololu Maestro. Same contract as `SimulatedDrive`.

    Constructing this centres both channels before returning, exactly as the
    PCA9685 backend does, so an ESC that arms on held neutral gets it for free
    and no code path can command motion through an uncentred channel.
    """

    def __init__(self, port: SerialPort,
                 channels: MaestroChannels | None = None) -> None:
        self._port = port
        self._ch = channels or MaestroChannels(throttle=Channel(0),
                                               steering=Channel(1))
        self._lock = threading.Lock()

        if self._ch.throttle.index == self._ch.steering.index:
            # Same refusal as the PCA9685 backend: one channel driving both an
            # ESC and a servo makes every steering input a throttle input, and
            # the first left turn is a launch.
            raise ValueError("throttle and steering cannot share a Maestro channel")

        if self._ch.steering_speed:
            self._command(_SET_SPEED, self._ch.steering.index,
                          *self._split(self._ch.steering_speed))
        self.neutral()

    # -- units ---------------------------------------------------------------

    @staticmethod
    def target(pulse_us: float) -> int:
        """Microseconds to the Maestro's quarter-microsecond target units.

        Clamped into the range the protocol can express in two 7-bit bytes, and
        NaN-safe, for the same reason every other conversion in this package is:
        an uninterpretable command must become a safe value rather than an
        arbitrary one, and on a wire where the high bit selects the command, an
        out-of-range number is not a wrong position but a different instruction.
        """
        if pulse_us != pulse_us:            # NaN
            return TARGET_OFF
        value = round(pulse_us * QUARTER_US)
        return int(max(0, min(0x3FFF, value)))

    @staticmethod
    def _split(value: int) -> tuple[int, int]:
        """Two 7-bit data bytes, low then high, bit 7 clear on both."""
        v = int(value) & 0x3FFF
        return v & 0x7F, (v >> 7) & 0x7F

    # -- the DriveHardware contract -----------------------------------------

    def set_drive(self, throttle: float, steering: float) -> None:
        with self._lock:
            self._write_channel(self._ch.steering, steering)
            # Throttle LAST on the way up: a failure partway through leaves the
            # car not-moving rather than moving-and-unsteerable.
            self._write_channel(self._ch.throttle, throttle)

    def neutral(self) -> None:
        with self._lock:
            try:
                self._write_channel(self._ch.throttle, 0.0)
            finally:
                # Runs even when stopping raised: a failed stop must not also
                # leave the wheels locked over for whatever happens next.
                self._write_channel(self._ch.steering, 0.0)

    def off(self) -> None:
        """Stop sending pulses on both channels.

        Distinct from neutral, and which one is correct depends on the ESC in
        front of it — some read signal-loss as a stop, some hold their last
        command. The runbook tests that on a stand rather than assuming it.
        """
        with self._lock:
            for channel in (self._ch.throttle, self._ch.steering):
                self._command(_SET_TARGET, channel.index, *self._split(TARGET_OFF))

    # -- diagnostics ---------------------------------------------------------

    def errors(self) -> int:
        """Read and CLEAR the Maestro's error flags.

        Bit 5 is the serial timeout. Reading it is the closest this code can get
        to observing the hardware failsafe — it can see the timeout FIRE, but it
        still cannot see whether the setting is enabled, so this is a diagnostic
        and not a check. Reading also clears the flags, which is the protocol's
        behaviour and not a choice made here.
        """
        with self._lock:
            self._command(_GET_ERRORS)
            data = self._port.read(2)
        if len(data) < 2:
            raise OSError("Maestro did not answer the error query — wrong serial "
                          "port (the command port is the lower-numbered ttyACM), "
                          "or the device is in USB Chained mode")
        return data[0] | (data[1] << 7)

    def go_home(self) -> None:
        """Send every channel to its configured startup position."""
        with self._lock:
            self._command(_GO_HOME)

    # -- wire ----------------------------------------------------------------

    def _write_channel(self, channel: Channel, value: float) -> None:
        target = self.target(channel.pulse_us(value))
        self._command(_SET_TARGET, channel.index, *self._split(target))

    def _command(self, command: int, *data: int) -> None:
        if self._ch.device_number is None:
            payload = bytes([command, *data])
        else:
            # Pololu protocol: 0xAA, device number, then the command with its
            # high bit stripped. Used when several controllers share a line.
            payload = bytes([0xAA, self._ch.device_number, command & 0x7F, *data])
        self._port.write(payload)


def open_serial_drive(port: str = "/dev/ttyACM0",
                      channels: MaestroChannels | None = None) -> MaestroDrive:
    """Construct a `MaestroDrive` on a real serial port.

    Kept out of `MaestroDrive.__init__` so the class never imports pyserial,
    which is what lets the tests run anywhere — the same code that will drive the
    car, exercised on a machine with nothing plugged in.
    """
    import serial  # imported lazily: hardware-only dependency

    # Baud rate is ignored on the Maestro's USB command port (it is a virtual
    # serial device), and is set only so pyserial has something to open with.
    # The timeout is NOT cosmetic: without it a read on a silent port blocks
    # forever, inside a lock, on the thread the deadman needs.
    return MaestroDrive(serial.Serial(port, 115200, timeout=0.2), channels=channels)
