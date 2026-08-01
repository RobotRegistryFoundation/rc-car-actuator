"""A budget for continuous motion, approved once.

The problem this solves: a phone driving a car sends commands at 20 Hz. Asking a
human to approve each one is absurd, and approving none of them means the
vehicle moves on nobody's authority. Neither is acceptable for a machine that
can roll into someone.

So the unit of approval is not a command, it is an ENVELOPE: a bounded budget of
motion that a human authorises once, and that every subsequent command draws
down. The car may move freely inside the budget and not at all outside it. One
signed decision covers a thousand commands without covering them indefinitely.

WHAT IS BOUNDED, and why each bound exists:

  motion_budget_s   Seconds the wheels may actually turn. The real currency: it
                    is what determines how far the vehicle can get.
  window_s          Wall-clock lifetime. Without it, an unused budget approved
                    this morning is still live tonight — authorisation must
                    decay even when nothing happens.
  max_throttle      Speed ceiling for this envelope. "Drive around the house"
                    and "drive in the yard" deserve different ceilings, and the
                    approver is the one who knows which they meant.

THE PROPERTY THAT MATTERS MOST: the budget only ever decreases. There is no
renew, no extend, no top-up. Getting more motion requires a new envelope, which
means a new approval. An envelope that could be extended by the thing spending
it would authorise nothing at all.

Charging is by ELAPSED MOTION, not by commands issued or leases granted.
Charging per lease would be catastrophic: at 20 Hz with a 0.5 s lease, a car
driving for one second would be billed ten. Charging per command would make the
budget depend on network chattiness rather than on how far the car went.
"""
from __future__ import annotations

import threading
import time
import uuid

#: Hard ceilings applied to any request, regardless of what was asked for. An
#: approval flow that can grant an unbounded budget is not a bound.
MAX_MOTION_BUDGET_S = 120.0
MAX_WINDOW_S = 600.0


class EnvelopeError(RuntimeError):
    """Motion was requested with no authority to move."""


class DriveEnvelope:
    """A bounded, single-approval budget for continuous motion.

    Thread-safe, and deliberately pure arithmetic: no I/O happens under the
    lock, so the stop path can charge the final segment without any risk of
    blocking. That distinction is load-bearing — the drive hardware lock was
    removed from the stop path for exactly this reason.
    """

    def __init__(self, motion_budget_s: float, window_s: float,
                 max_throttle: float, approved_by: str = "") -> None:
        self.id = f"env-{uuid.uuid4().hex[:12]}"
        self.motion_budget_s = min(abs(float(motion_budget_s)), MAX_MOTION_BUDGET_S)
        self.window_s = min(abs(float(window_s)), MAX_WINDOW_S)
        self.max_throttle = min(abs(float(max_throttle)), 1.0)
        self.approved_by = approved_by

        self.opened_at = time.monotonic()
        self.expires_at = self.opened_at + self.window_s
        self._spent_s = 0.0
        #: When the current stretch of motion began, or None while stopped.
        #: Charging happens when a stretch ENDS, so idling costs nothing.
        self._motion_since: float | None = None
        self._revoked = False
        self._lock = threading.Lock()

    # -- state -------------------------------------------------------------

    @property
    def spent_s(self) -> float:
        """Motion charged so far, including any stretch still running."""
        with self._lock:
            return self._spent_s + self._open_segment_locked()

    def _open_segment_locked(self) -> float:
        if self._motion_since is None:
            return 0.0
        return max(0.0, time.monotonic() - self._motion_since)

    @property
    def remaining_s(self) -> float:
        return max(0.0, self.motion_budget_s - self.spent_s)

    @property
    def revoked(self) -> bool:
        return self._revoked

    @property
    def expired(self) -> bool:
        """True once this envelope can no longer authorise motion, for any reason."""
        return (self._revoked
                or time.monotonic() >= self.expires_at
                or self.remaining_s <= 0.0)

    def reason_unusable(self) -> str | None:
        """Why motion is refused, in the approver's terms. None when usable."""
        if self._revoked:
            return "the drive approval was revoked"
        if time.monotonic() >= self.expires_at:
            return (f"the drive approval expired "
                    f"({self.window_s:.0f}s window elapsed)")
        if self.remaining_s <= 0.0:
            return (f"the drive approval is used up "
                    f"({self.motion_budget_s:.1f}s of motion spent)")
        return None

    # -- charging ----------------------------------------------------------

    def begin_motion(self) -> None:
        """Mark the start of a stretch of motion. Idempotent while moving."""
        with self._lock:
            if self._motion_since is None:
                self._motion_since = time.monotonic()

    def end_motion(self) -> float:
        """Close the current stretch and charge it. Returns seconds charged.

        Safe to call when not moving (charges nothing), because the stop path
        cannot know whether the car was moving and must not have to ask.
        """
        with self._lock:
            charged = self._open_segment_locked()
            self._spent_s += charged
            self._motion_since = None
            return charged

    def revoke(self) -> None:
        """End this envelope's authority immediately and permanently."""
        self._revoked = True
        self.end_motion()

    def cap_throttle(self, throttle: float) -> float:
        """Apply this envelope's speed ceiling."""
        return max(-self.max_throttle, min(self.max_throttle, throttle))

    def describe(self) -> dict:
        """Telemetry. Carries the id so a receipt traces to its approval."""
        return {
            "envelope_id": self.id,
            "approved_by": self.approved_by,
            "motion_budget_s": round(self.motion_budget_s, 3),
            "motion_spent_s": round(self.spent_s, 3),
            "motion_remaining_s": round(self.remaining_s, 3),
            "window_s": round(self.window_s, 1),
            "window_remaining_s": round(max(0.0, self.expires_at - time.monotonic()), 1),
            "max_throttle": self.max_throttle,
            "revoked": self._revoked,
            "usable": not self.expired,
            "unusable_because": self.reason_unusable(),
        }
