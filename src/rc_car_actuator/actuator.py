"""RCAN actuator for a wire-controlled RC car.

A moving vehicle is a different safety problem from an arm. The arm's `move()`
polls until the joint arrives and returns when it is there: motion is bounded
because the target is a POSITION. A car has no target position. Ask it for
throttle and it keeps that throttle until something says otherwise, and "until
something says otherwise" is exactly the part that fails when a phone locks or
Wi-Fi drops.

So motion here is a LEASE, not a command:

    execute() writes the setpoint, extends the lease, and RETURNS IMMEDIATELY.

`duration_s` is how long the lease lasts, NOT how long execute() sleeps. This is
the single most important design decision in the file. If execute() slept for
the duration it would hold the request path for the entire motion, and the very
next request — the one saying "stop" — would queue behind it. The stop would
arrive after the motion it was meant to cancel. Motion ends because the lease
expires on the deadman's own thread, not because anyone waited for it.

WHAT THIS DOES NOT PROTECT AGAINST, stated plainly: the deadman is a Python
thread on Linux. It covers a locked phone, a crashed app, dropped Wi-Fi, a hung
gateway. It does NOT cover the kernel stalling, this process being SIGKILLed, or
the Pi browning out — and a brownout is most likely exactly when the motor draws
current. The authoritative stop must be a lease that expires in FIRMWARE, on an
MCU between the Pi and the ESC, which survives Linux dying entirely. Until that
exists this vehicle belongs on a stand with its wheels off the ground.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from robot_md_gateway.actuator import ActuatorOutcome

from rc_car_actuator.deadman import DEFAULT_TIMEOUT_S, Deadman
from rc_car_actuator.drive import (
    DEFAULT_MAX_THROTTLE,
    DriveHardware,
    SimulatedDrive,
    clamp,
)

logger = logging.getLogger("rc_car.actuator")

#: One caller at a time on the drive hardware. The gateway serves requests from
#: a threadpool, and two overlapping drive commands interleaving their writes
#: would leave the ESC holding whichever value happened to land last.
_DRIVE_LOCK = threading.Lock()

#: The longest a single command may keep the car alive without being renewed.
#: A lease is not a schedule: asking for 30 seconds of throttle and walking away
#: is precisely the thing this design exists to prevent.
MAX_LEASE_S = 2.0

#: ROBOT.md capability names this driver actually implements.
IMPLEMENTED_CAPABILITIES: frozenset[str] = frozenset({
    "drive.set", "drive.stop", "status.report",
})

#: Minimum caller tiers per tool, enforced here as defence in depth. The
#: gateway's own gate keys off the envelope's self-declared `scope`, which the
#: CALLER controls; this keys off the tool name, which the OPERATOR controls.
#:
#: `drive.stop` is deliberately the most permissive thing in the file: anyone who
#: can reach this driver at all may stop the car. A stop that needed a privilege
#: check is a stop that can be refused, and there is no failure mode where
#: refusing to stop a moving vehicle is the safer choice.
REQUIRED_TIERS: dict[str, frozenset[str]] = {
    "drive.set": frozenset({"actuate", "commission"}),
    "drive.stop": frozenset({"read", "actuate", "commission"}),
    "status.report": frozenset({"read", "actuate", "commission"}),
}


class RCCarActuator:
    """RobotRegistryFoundation/rc-car-actuator — wire-controlled RC car."""

    name = "rc-car"
    description = "Wire-controlled RC car drive actuator with a heartbeat deadman."
    config_schema: dict = {}

    capabilities = ("drive.set", "drive.stop", "status.report")

    def __init__(
        self,
        hardware: DriveHardware | None = None,
        max_throttle: float = DEFAULT_MAX_THROTTLE,
        lease_timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        """
        Args:
            hardware: The drive layer. Defaults to `SimulatedDrive`, which moves
                nothing — a driver constructed with no arguments must not be
                able to move a real vehicle by accident.
            max_throttle: Ceiling applied to every commanded throttle, on top of
                whatever the caller asked for.
        """
        self._hw: DriveHardware = hardware if hardware is not None else SimulatedDrive()
        self._max_throttle = abs(clamp(max_throttle))
        self._last_command: tuple[float, float] = (0.0, 0.0)
        self._commands = 0
        self._estopped = False

        # The deadman owns stopping. It starts EXPIRED, so the car cannot move
        # until a command actually arrives.
        self._deadman = Deadman(stop=self._stop_hardware, timeout_s=lease_timeout_s)

    # -- hardware ----------------------------------------------------------

    def _stop_hardware(self) -> None:
        """Bring the vehicle to neutral. Idempotent; called on a timer.

        DELIBERATELY DOES NOT TAKE `_DRIVE_LOCK`. If a command path were wedged
        while holding it — a hung pigpio socket, a hardware layer that blocks —
        then a stop that waited for that lock would wait forever, and the car
        would keep driving on the last throttle it was given. The stop path must
        never be blockable by the command path.

        Skipping the lock is safe because the lock guards ACTUATOR STATE, not the
        wire: each hardware implementation serialises its own writes internally,
        so a neutral issued here is still atomic at the hardware layer.
        """
        try:
            self._hw.neutral()
        finally:
            self._last_command = (0.0, 0.0)

    # -- capabilities ------------------------------------------------------

    def drive_set(self, throttle: float, steering: float, duration_s: float) -> dict:
        """Set the drive setpoint and extend the lease. DOES NOT BLOCK.

        Returns as soon as the setpoint is written — typically about a
        millisecond. The car keeps moving because the lease is alive, and stops
        when it expires. Nothing here sleeps for `duration_s`; see the module
        docstring for why that distinction is the whole design.
        """
        if self._estopped:
            raise RuntimeError(
                "e-stop is engaged; clear it before commanding motion")

        lease = max(0.0, min(MAX_LEASE_S, float(duration_s)))
        if lease <= 0:
            # A zero-length lease is a stop, not a no-op. Treating it as "ignore"
            # would leave the previous throttle running.
            self.drive_stop()
            return {"throttle": 0.0, "steering": 0.0, "lease_s": 0.0,
                    "note": "zero duration treated as stop"}

        applied_throttle = clamp(throttle, self._max_throttle)
        applied_steering = clamp(steering)

        with _DRIVE_LOCK:
            self._hw.set_drive(applied_throttle, applied_steering)
            self._last_command = (applied_throttle, applied_steering)
            self._commands += 1

        # Fed AFTER the write succeeds. Feeding first would keep the car alive
        # on the strength of a command that then failed to reach the hardware.
        self._deadman.feed()

        # Re-check, because the check at the top of this method raced: an e-stop
        # arriving while the throttle was being written would otherwise have its
        # neutral overwritten by this command, and the feed above would re-arm
        # the lease — a car driving away from a pressed emergency stop.
        #
        # Checking again after the fact closes that window from this side, so
        # e-stop needs no lock and can never deadlock against a hardware write.
        if self._estopped:
            self.drive_stop()
            raise RuntimeError("e-stop engaged while the command was being applied")

        return {
            "throttle": applied_throttle,
            "steering": applied_steering,
            "requested_throttle": float(throttle),
            "throttle_capped": abs(clamp(throttle)) > self._max_throttle,
            "lease_s": lease,
            # Said explicitly because it is the counterintuitive part: the call
            # has returned, and the car is still moving.
            "blocking": False,
            "stops_at_monotonic": time.monotonic() + lease,
        }

    def drive_stop(self) -> dict:
        """Stop now. Always permitted, always safe to call."""
        self._deadman.expire_now()
        return {"stopped": True, "throttle": 0.0, "steering": 0.0}

    def estop(self) -> dict:
        """Stop and REFUSE further motion until explicitly cleared.

        Distinct from `drive_stop`: a stop that the next command can immediately
        undo is not an emergency stop.
        """
        # Flag first, then stop. In this order a command that is mid-flight
        # observes the flag on its post-write re-check and stops itself; the
        # reverse order would let that command's throttle land after this
        # neutral. Takes no lock, so it can never wait on the command path.
        self._estopped = True
        self._deadman.expire_now()
        return {"estopped": True}

    def clear_estop(self) -> dict:
        """Release the e-stop. Does NOT resume motion — the lease is still expired."""
        self._estopped = False
        return {"estopped": False, "note": "cleared; the car remains stopped until commanded"}

    def read_state(self) -> dict:
        throttle, steering = self._last_command
        return {
            "throttle": throttle,
            "steering": steering,
            "moving": self._deadman.alive and throttle != 0.0,
            "lease_alive": self._deadman.alive,
            "lease_seconds_remaining": round(self._deadman.seconds_remaining, 3),
            "estopped": self._estopped,
            "max_throttle": self._max_throttle,
            "commands_accepted": self._commands,
            "hardware": type(self._hw).__name__,
            # The honest caveat, carried in telemetry so it reaches anyone
            # reading state rather than only anyone reading the source.
            "stop_layers": ["software deadman (this process)"],
            "firmware_lease": False,
        }

    def shutdown(self) -> None:
        """Stop the watchdog and the vehicle."""
        self._deadman.shutdown()

    # -- gateway entry point -----------------------------------------------

    def execute(
        self,
        *,
        envelope: dict,
        manifest_path: Path,
        tier: str,
        config: dict,
    ) -> ActuatorOutcome:
        """Dispatch an RCAN INVOKE envelope. Never blocks for the motion."""
        tool_name = envelope.get("tool_name")
        tool_args = envelope.get("tool_args", {}) or {}

        # Tier is re-checked HERE against the TOOL, not against the envelope's
        # self-declared `scope`, which the caller supplies and can simply lie
        # about. The two gates are independent on purpose.
        required = REQUIRED_TIERS.get(tool_name)
        if required is not None and tier not in required:
            return ActuatorOutcome(
                success=False,
                outcome_kind="denied",
                error_message=(f"tier {tier!r} may not invoke {tool_name!r} "
                               f"(requires one of {sorted(required)})"),
            )

        try:
            if tool_name == "drive.set":
                telemetry = self.drive_set(
                    throttle=tool_args.get("throttle", 0.0),
                    steering=tool_args.get("steering", 0.0),
                    duration_s=tool_args.get("duration_s", 0.0),
                )
            elif tool_name == "drive.stop":
                telemetry = self.drive_stop()
            elif tool_name == "status.report":
                telemetry = self.read_state()
            else:
                return ActuatorOutcome(
                    success=False,
                    outcome_kind="error",
                    error_message=f"unknown capability: {tool_name!r}",
                )
        except Exception as exc:
            # A failure while commanding motion must not leave the car moving on
            # a lease that a partially-applied command already extended.
            self.drive_stop()
            return ActuatorOutcome(
                success=False,
                outcome_kind="error",
                error_message=f"{type(exc).__name__}: {exc}",
            )

        return ActuatorOutcome(
            success=True,
            outcome_kind="executed",
            telemetry=telemetry,
        )
