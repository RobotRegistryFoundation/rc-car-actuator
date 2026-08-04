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

AND THE RECEIPT MUST SAY WHEN THAT IS. Every field this file returns is signed by
the gateway and kept as evidence, so `lease_s` and `stops_at_monotonic` are not
descriptions of intent — they are claims about a physical vehicle that someone
will later rely on. They are therefore read back from the deadman AFTER it has
granted the lease, never computed alongside it. This driver once reported a 2.0 s
lease while a fixed 0.4 s timeout stopped the car: safe by luck, and a signed
untruth about when the wheels stop. The number in the receipt is now the number
the watchdog thread enforces, clamped by the same two bounds — the per-command
ceiling and the envelope's remaining motion budget — that actually apply.

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
from pathlib import Path

from robot_md_gateway.actuator import ActuatorOutcome

from rc_car_actuator.deadman import DEFAULT_TIMEOUT_S, MAX_LEASE_S, Deadman
from rc_car_actuator.envelope import DriveEnvelope, EnvelopeError
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

# MAX_LEASE_S — the longest a single command may keep the car alive without
# being renewed — is re-exported from `deadman`, where it now lives. A lease is
# not a schedule: asking for 30 seconds of throttle and walking away is
# precisely the thing this design exists to prevent, and a bound stated here
# while the watchdog enforced a different one is precisely how this driver came
# to sign receipts it did not honour. One ceiling, in one place, inside the
# thread that applies it.
__all__ = ["MAX_LEASE_S", "RCCarActuator", "IMPLEMENTED_CAPABILITIES",
           "REQUIRED_TIERS"]

#: ROBOT.md capability names this driver actually implements.
IMPLEMENTED_CAPABILITIES: frozenset[str] = frozenset({
    "drive.set", "drive.stop", "status.report",
    "drive.envelope.open", "drive.envelope.revoke",
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
    # Opening an envelope is the approval itself — it is what grants a budget of
    # physical motion — so it demands the same tier as moving.
    "drive.envelope.open": frozenset({"actuate", "commission"}),
    # Revoking is like stopping: never refuse it. Someone who can see the car
    # can withdraw its permission to move.
    "drive.envelope.revoke": frozenset({"read", "actuate", "commission"}),
}


class RCCarActuator:
    """RobotRegistryFoundation/rc-car-actuator — wire-controlled RC car."""

    name = "rc-car"
    description = "Wire-controlled RC car drive actuator with a heartbeat deadman."
    config_schema: dict = {}

    capabilities = ("drive.set", "drive.stop", "status.report",
                    "drive.envelope.open", "drive.envelope.revoke")

    def __init__(
        self,
        hardware: DriveHardware | None = None,
        max_throttle: float = DEFAULT_MAX_THROTTLE,
        lease_timeout_s: float = DEFAULT_TIMEOUT_S,
        max_lease_s: float = MAX_LEASE_S,
    ) -> None:
        """
        Args:
            hardware: The drive layer. Defaults to `SimulatedDrive`, which moves
                nothing — a driver constructed with no arguments must not be
                able to move a real vehicle by accident.
            max_throttle: Ceiling applied to every commanded throttle, on top of
                whatever the caller asked for.
            lease_timeout_s: Lease given to a command that states no duration.
            max_lease_s: Longest lease any single command may buy. An operator
                may TIGHTEN this — a tow-test wants half a second, not two — but
                the deadman clamps it to `MAX_LEASE_S` regardless, so nothing
                here can widen the bound.
        """
        self._hw: DriveHardware = hardware if hardware is not None else SimulatedDrive()
        self._max_throttle = abs(clamp(max_throttle))
        self._last_command: tuple[float, float] = (0.0, 0.0)
        self._commands = 0
        self._estopped = False
        #: The current approval. None means the car has no authority to move at
        #: all — the default, and the state it returns to when a budget runs out.
        self._envelope: DriveEnvelope | None = None

        # The deadman owns stopping. It starts EXPIRED, so the car cannot move
        # until a command actually arrives.
        self._deadman = Deadman(stop=self._stop_hardware, timeout_s=lease_timeout_s,
                                max_lease_s=max_lease_s)

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
            # Close the billing segment here too, not only on an explicit stop:
            # the common case is a lease expiring on the deadman's own thread
            # with nobody calling anything. Charging only on explicit stops
            # would make every abandoned command free, and abandonment is
            # exactly what the deadman exists to handle.
            #
            # Pure arithmetic under a short lock — no I/O — so this cannot wedge
            # the stop path the way the hardware lock once did.
            if self._envelope is not None:
                self._envelope.end_motion()

    # -- capabilities ------------------------------------------------------

    def drive_set(self, throttle: float, steering: float,
                  duration_s: float | None = None) -> dict:
        """Set the drive setpoint and extend the lease. DOES NOT BLOCK.

        Returns as soon as the setpoint is written — typically about a
        millisecond. The car keeps moving because the lease is alive, and stops
        when it expires. Nothing here sleeps for `duration_s`; see the module
        docstring for why that distinction is the whole design.

        `duration_s=None` means "no opinion" and takes the deadman's short
        default. That is NOT the same as the RCAN contract's `duration_s: 0`,
        which is an explicit request for a zero-length lease — i.e. a stop.
        """
        if self._estopped:
            raise RuntimeError(
                "e-stop is engaged; clear it before commanding motion")

        # The ceiling is applied by the deadman, which is what actually enforces
        # it. Asking it rather than re-deriving the bound here is the difference
        # between a receipt that describes the mechanism and a receipt that
        # describes a second, hopeful copy of the mechanism.
        requested_lease = None if duration_s is None else float(duration_s)
        lease = self._deadman.clamp_lease(duration_s)
        lease_cut_by = None
        if requested_lease is not None and lease < requested_lease:
            lease_cut_by = "ceiling"
        if lease <= 0:
            # A zero-length lease is a stop, not a no-op. Treating it as "ignore"
            # would leave the previous throttle running.
            self.drive_stop()
            return {"throttle": 0.0, "steering": 0.0, "lease_s": 0.0,
                    "note": "zero duration treated as stop"}

        envelope = self._envelope
        if envelope is None:
            raise EnvelopeError(
                "no drive approval is open — approve a drive budget before moving")
        unusable = envelope.reason_unusable()
        if unusable is not None:
            # Stop as well as refuse. The car may still be rolling on a lease
            # granted moments ago, and a refusal that leaves it moving is not a
            # refusal of motion.
            self.drive_stop()
            raise EnvelopeError(unusable)

        # A lease may never outlive the budget that pays for it. Without this a
        # 2 s lease granted with 0.5 s of budget remaining would drive for the
        # full 2 s — the budget would be checked at command time and then
        # ignored while the car was actually moving, which is the only time it
        # matters. Capping here means the deadman, which already stops the car
        # when a lease ends, also stops it when the budget runs out. One
        # mechanism, not two racing ones.
        if envelope.remaining_s < lease:
            lease = envelope.remaining_s
            # The budget is the tighter bound, and says so: "you asked for 2 s
            # and got 0.3 s" is only actionable if the receipt names which limit
            # did the cutting.
            lease_cut_by = "budget"

        # The envelope's ceiling composes with the driver's own; the lower wins.
        applied_throttle = clamp(throttle, self._max_throttle)
        applied_throttle = envelope.cap_throttle(applied_throttle)
        applied_steering = clamp(steering)

        with _DRIVE_LOCK:
            self._hw.set_drive(applied_throttle, applied_steering)
            self._last_command = (applied_throttle, applied_steering)
            self._commands += 1

        # Billing starts only once the wheels are actually commanded to turn,
        # and only for real motion — a steering-only command with zero throttle
        # moves the car nowhere and costs nothing.
        if applied_throttle != 0.0:
            envelope.begin_motion()
        else:
            envelope.end_motion()

        # Fed AFTER the write succeeds. Feeding first would keep the car alive
        # on the strength of a command that then failed to reach the hardware.
        #
        # `granted` is what the deadman will actually enforce, and it — not
        # `duration_s`, and not `lease` — is what the receipt below reports. The
        # two used to differ by 5x: receipts said 2.0 s while a fixed 0.4 s
        # timeout stopped the car. It erred safe and it was still a lie, and a
        # signed lie about when a vehicle stops is worse than no signature.
        granted = self._deadman.feed(for_s=lease)
        stops_at = self._deadman.expires_at_monotonic
        if stops_at is None:
            # A stop, an e-stop or a revoke landed on another thread between the
            # feed and this read. The car is neutral, so the receipt says the
            # lease is gone rather than quoting one that no longer exists. (The
            # re-checks below usually turn this into an outright refusal; this
            # keeps the telemetry honest even when they do not.)
            granted = 0.0
            lease_cut_by = "stopped"

        # Re-check EVERYTHING that can withdraw permission, because the checks at
        # the top of this method raced. An e-stop or a revocation arriving while
        # the throttle was being written would otherwise have its neutral
        # overwritten by this command, and the feed above would re-arm the lease
        # — a car driving away from a pressed emergency stop, or driving on
        # authority that was explicitly taken away.
        #
        # Checking again after the fact closes that window from this side, which
        # is why neither e-stop nor revoke needs a lock and why neither can
        # deadlock against a hardware write.
        if self._estopped:
            self.drive_stop()
            raise RuntimeError("e-stop engaged while the command was being applied")
        withdrawn = envelope.reason_unusable()
        if withdrawn is not None or self._envelope is not envelope:
            self.drive_stop()
            raise EnvelopeError(
                withdrawn or "the drive approval was replaced while the command was "
                             "being applied")

        return {
            "throttle": applied_throttle,
            "steering": applied_steering,
            "requested_throttle": float(throttle),
            "throttle_capped": abs(clamp(throttle)) > self._max_throttle,
            # The lease GRANTED, read back from the deadman that will enforce
            # it — never the lease requested.
            "lease_s": granted,
            "requested_lease_s": requested_lease,
            # Which bound cut the request, in the approver's terms, so a short
            # lease is legible instead of merely surprising.
            "lease_cut_by": lease_cut_by,
            # Said explicitly because it is the counterintuitive part: the call
            # has returned, and the car is still moving.
            "blocking": False,
            "stops_at_monotonic": stops_at,
            # Expiry is detected on a polling tick, so the wheels go neutral in
            # [stops_at, stops_at + this]. Stated rather than implied: promising
            # a stop instant to the microsecond would be a second untruth on the
            # same receipt.
            "stop_detect_within_s": self._deadman.tick_s,
            # Carried on every motion receipt so a signed record of the car
            # moving traces back to the approval that permitted it. Without this
            # the gateway's journal shows a thousand drive commands and no way to
            # tell which human decision authorised them.
            "envelope_id": envelope.id,
            "motion_remaining_s": round(envelope.remaining_s, 3),
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

    def envelope_open(self, motion_budget_s: float, window_s: float,
                      max_throttle: float, approved_by: str = "") -> dict:
        """Approve a bounded budget of motion.

        Opening a new envelope REPLACES any current one rather than adding to
        it, and closes out the old one's billing first. Accumulating budgets
        would make "approve a little more" indistinguishable from "approve
        without limit" after enough repetitions.
        """
        if self._estopped:
            raise RuntimeError("e-stop is engaged; clear it before approving motion")
        if self._envelope is not None:
            self._envelope.end_motion()
        # Any motion authorised by the previous envelope ends here, so the new
        # budget starts from a stopped car rather than inheriting a live lease.
        self.drive_stop()
        self._envelope = DriveEnvelope(
            motion_budget_s=motion_budget_s,
            window_s=window_s,
            max_throttle=max_throttle,
            approved_by=approved_by,
        )
        return self._envelope.describe()

    def envelope_revoke(self) -> dict:
        """Withdraw the current approval and stop the car."""
        if self._envelope is None:
            return {"revoked": False, "note": "no drive approval was open"}
        self._envelope.revoke()
        # Revoking authority without stopping the vehicle would be theatre.
        self.drive_stop()
        described = self._envelope.describe()
        self._envelope = None
        return {"revoked": True, **described}

    def read_state(self) -> dict:
        throttle, steering = self._last_command
        return {
            "throttle": throttle,
            "steering": steering,
            "moving": self._deadman.alive and throttle != 0.0,
            "lease_alive": self._deadman.alive,
            # The lease in force RIGHT NOW, so a reader can check a receipt's
            # claim against the mechanism without waiting for it to expire.
            "lease_s": round(self._deadman.lease_s, 3),
            "lease_seconds_remaining": round(self._deadman.seconds_remaining, 3),
            "lease_max_s": self._deadman.max_lease_s,
            "estopped": self._estopped,
            "max_throttle": self._max_throttle,
            "commands_accepted": self._commands,
            "hardware": type(self._hw).__name__,
            # The honest caveat, carried in telemetry so it reaches anyone
            # reading state rather than only anyone reading the source.
            "stop_layers": ["software deadman (this process)"],
            "firmware_lease": False,
            "envelope": self._envelope.describe() if self._envelope else None,
            "may_move": (self._envelope is not None
                         and not self._envelope.expired
                         and not self._estopped),
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
            elif tool_name == "drive.envelope.open":
                telemetry = self.envelope_open(
                    motion_budget_s=tool_args.get("motion_budget_s", 0.0),
                    window_s=tool_args.get("window_s", 0.0),
                    max_throttle=tool_args.get("max_throttle", self._max_throttle),
                    approved_by=tool_args.get("approved_by", ""),
                )
            elif tool_name == "drive.envelope.revoke":
                telemetry = self.envelope_revoke()
            elif tool_name == "status.report":
                telemetry = self.read_state()
            else:
                return ActuatorOutcome(
                    success=False,
                    outcome_kind="error",
                    error_message=f"unknown capability: {tool_name!r}",
                )
        except EnvelopeError as exc:
            # Refusing to move without an approval is a DECISION, not a fault, and
            # the two must not arrive at the caller looking alike. Reported as
            # `denied` so the gateway signs it into a 403 receipt the operator can
            # verify and keep — "the car would not move, and here is the signed
            # reason" — rather than a 500 that reads as the robot falling over.
            #
            # Still stop first: the car may be rolling on a lease granted moments
            # ago, and a refusal that leaves it moving is not a refusal of motion.
            self.drive_stop()
            return ActuatorOutcome(
                success=False,
                outcome_kind="denied",
                error_message=str(exc),
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
