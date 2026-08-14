"""The wire between a command and the wheels.

Two implementations, and the split matters:

`SimulatedDrive` records what it was told and moves nothing. Every test in this
package runs against it, so the safety behaviour — neutral on construction,
neutral on expiry, bounded throttle — is genuinely exercised rather than
asserted.

`PWMDrive` talks to a real ESC and steering servo over hardware PWM. It is
UNVERIFIED: no car has been wired to this Pi yet. It is written to be obviously
correct rather than clever, and its numbers (pulse widths, channel numbers) are
the standard hobby-RC values that must be checked against the actual ESC before
anything touches the ground.

THE ONE RULE THAT SHAPES THIS FILE: a PWM peripheral keeps emitting its last
value forever, with no further CPU involvement. Writing "go" once means "go
forever" unless something else intervenes. So every path that can fail lands on
neutral, and neutral is written repeatedly rather than once.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger("rc_car.drive")

#: Throttle and steering are unitless, -1.0 to +1.0. Deliberately NOT metres per
#: second: this vehicle has no speed sensor, so a value in m/s would be a
#: measurement the hardware cannot make.
FULL_SCALE = 1.0

#: Default ceiling on commanded throttle. Well under full scale because an
#: untested vehicle's first motion should be a crawl, and because the deadman's
#: 400 ms lease is only a short distance at low speed.
DEFAULT_MAX_THROTTLE = 0.35


class DriveHardware(Protocol):
    """The minimum a drive layer must do."""

    def set_drive(self, throttle: float, steering: float) -> None: ...
    def neutral(self) -> None: ...


def clamp(value: float, limit: float = FULL_SCALE) -> float:
    """Constrain to +/-limit, mapping NaN to 0.

    NaN is handled explicitly because it fails every comparison silently:
    `min(max(nan, -1), 1)` returns nan, which would reach the PWM layer and
    become an undefined pulse width. A command nobody can interpret must become
    the safe value, not an arbitrary one.
    """
    if value != value:  # NaN
        return 0.0
    return max(-limit, min(limit, float(value)))


@dataclass(frozen=True)
class Channel:
    """One PWM output, and the numbers that make it mean something.

    Lives in this module rather than beside a particular driver because it is
    the same three measurements whatever chip is emitting the pulse: a PCA9685
    and a Pololu Maestro disagree about registers and protocols and agree
    completely about what "neutral" means.

    A channel is not just a number, because two identical-looking servo headers
    on the same board disagree about what centre is. The ESC's neutral is
    whatever it was taught during its calibration; the steering servo's centre
    is wherever the linkage happens to put the wheels straight. Both are
    per-vehicle measurements, and hardcoding 1500 us for both is how a car
    creeps forward at rest and tracks 5 degrees left.
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


class SimulatedDrive:
    """Records commands. Moves nothing. The default, and what tests use."""

    def __init__(self) -> None:
        self.history: list[tuple[float, float]] = []
        self._lock = threading.Lock()
        # Starts neutral, and records it — the same claim the real hardware
        # makes on construction.
        self.neutral()

    @property
    def last(self) -> tuple[float, float]:
        with self._lock:
            return self.history[-1]

    @property
    def neutral_count(self) -> int:
        with self._lock:
            return sum(1 for t, s in self.history if t == 0.0)

    def set_drive(self, throttle: float, steering: float) -> None:
        with self._lock:
            self.history.append((clamp(throttle), clamp(steering)))

    def neutral(self) -> None:
        with self._lock:
            self.history.append((0.0, 0.0))


class PWMDrive:
    """Hardware PWM to a hobby ESC and steering servo.

    UNVERIFIED — no vehicle has been connected. Check every number here against
    the actual ESC's documentation before the wheels touch the ground, and do
    the first run with the car on a stand.

    Pulse widths are the hobby-RC standard: 1000 us full reverse, 1500 us
    neutral, 2000 us full forward. Many ESCs also require an arming sequence
    (neutral held for a second or two at power-on) before they respond at all;
    holding neutral from construction satisfies that for free.
    """

    #: Microseconds. 1500 is neutral for both the ESC and the steering servo.
    PULSE_NEUTRAL_US = 1500
    PULSE_SPAN_US = 500

    def __init__(self, pi, throttle_gpio: int, steering_gpio: int) -> None:  # noqa: ANN001
        """
        Args:
            pi: A pigpio.pi() connection, or anything with the same
                `set_servo_pulsewidth(gpio, microseconds)` method.
        """
        self._pi = pi
        self._throttle_gpio = throttle_gpio
        self._steering_gpio = steering_gpio
        self._lock = threading.Lock()
        # Neutral BEFORE anything else can command motion. If this raises, the
        # object does not exist and nothing can drive through it.
        self.neutral()

    def _pulse(self, value: float) -> int:
        return int(self.PULSE_NEUTRAL_US + clamp(value) * self.PULSE_SPAN_US)

    def set_drive(self, throttle: float, steering: float) -> None:
        with self._lock:
            self._pi.set_servo_pulsewidth(self._steering_gpio, self._pulse(steering))
            # Throttle written LAST on the way up, so a failure partway through
            # leaves the car not-moving rather than moving-and-unsteerable.
            self._pi.set_servo_pulsewidth(self._throttle_gpio, self._pulse(throttle))

    def neutral(self) -> None:
        with self._lock:
            # Throttle FIRST on the way down: stopping matters more than
            # straightening, and if the second write fails the car is already
            # stopped.
            try:
                self._pi.set_servo_pulsewidth(self._throttle_gpio, self.PULSE_NEUTRAL_US)
            finally:
                self._pi.set_servo_pulsewidth(self._steering_gpio, self.PULSE_NEUTRAL_US)
