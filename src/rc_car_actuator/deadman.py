"""The deadman: a moving vehicle must stop when commands stop.

This is the single most important file in the package. Everything else decides
where the car goes; this decides that it stops when nobody is steering it.

The failure it exists to prevent is not exotic. A phone locks, an app crashes,
Wi-Fi drops behind a fridge, the gateway process hangs on a slow request — and
an ESC holds its last throttle command **forever**, because PWM peripherals keep
emitting the last value written to them with no further CPU involvement. That is
a car driving into a wall at the last speed anyone asked for.

So the rule is inverted: throttle is not something you set, it is something you
must keep renewing. Every renewal buys a bounded LEASE — the one it asked for,
cut to a hard ceiling — and when that lease runs out the vehicle returns to
neutral whether or not anyone is listening.

Design constraints that follow from that, and why:

  * It runs in its OWN thread with no dependency on the request path. If the
    gateway blocks, this keeps running.
  * It writes neutral REPEATEDLY once expired, not once. A single write can be
    lost to an I2C glitch, and "we told it to stop once" is not a safety story.
  * It starts EXPIRED. On boot, before any command has ever arrived, the safe
    value is neutral — not "whatever the PWM chip powered up holding".
  * `feed()` is the only thing that keeps the car alive, and it is called on the
    ACTUATION path only. Reading telemetry must never feed it.

WHERE THIS IS NOT ENOUGH — read before trusting it:

This is a Python thread on Linux. It covers the common failures (phone locks,
app crashes, Wi-Fi drops, gateway hangs) because it does not share a thread with
the request path. It does NOT cover the kernel stalling, the process being
SIGKILLed, or the Pi browning out mid-drive — and a brownout is likely precisely
when the motor draws current.

So this is the SOFTWARE layer of a two-layer stop. The authoritative layer must
be a lease that expires in firmware: an MCU between the Pi and the ESC that
returns to neutral on its own timer unless the Pi keeps refreshing it. That one
survives Linux dying entirely. Until it exists, this vehicle should only run
with its wheels off the ground.

A consequence for the actuator that uses this: `execute()` MUST NOT BLOCK. The
arm's move() polls to convergence and returns when the joint arrives; a drive
command that slept for its duration would hold the bus and delay the very stop
that makes it safe. A drive duration is a LEASE, not a sleep — write the
setpoint, extend the lease, return in about a millisecond. Motion ends because
the lease expires, not because anyone waited for it.

THE LEASE IS PER-FEED, AND THAT IS A CORRECTNESS REQUIREMENT, NOT A FEATURE:

This class used to hold a single fixed timeout, so `feed()` always bought
`DEFAULT_TIMEOUT_S` no matter what the caller had asked for or been told it got.
The actuator meanwhile signed receipts saying `lease_s: 2.0`. Both numbers were
defensible on their own and the disagreement erred safe — the car stopped early
— but a signed receipt that misstates when the wheels stop is exactly the
evidence this project exists to make impossible. A receipt is only worth
something if the mechanism it describes is the mechanism that runs.

So `feed(for_s=...)` sets the expiry for THAT feed, and returns the lease it
actually granted. Whoever writes the receipt writes down the return value, not
the request. Two bounds apply, both inside this class so no caller can talk its
way past them:

  * `max_lease_s` (MAX_LEASE_S) is a hard ceiling on any single feed. A lease is
    not a schedule; asking for thirty seconds of throttle and walking away is
    the failure this whole file exists to prevent.
  * A feed with no duration at all gets `timeout_s` (DEFAULT_TIMEOUT_S), the
    short conservative default. Silence about duration must never buy the
    ceiling.

A shorter lease than requested is always allowed — every clamp here only ever
cuts. `expire_now()` outranks any live lease, however long.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

logger = logging.getLogger("rc_car.deadman")

#: How long a command with NO stated duration keeps the vehicle alive.
#:
#: Short enough that a stall is felt almost immediately at walking pace, long
#: enough to ride out ordinary Wi-Fi jitter without stuttering. At 1 m/s a
#: 400 ms lapse is ~40 cm of unattended travel, which is the most that should
#: ever happen when nobody said how long they wanted.
DEFAULT_TIMEOUT_S = 0.4

#: The longest any single feed may keep the vehicle alive without being renewed,
#: whatever the caller asks for.
#:
#: 2.0 s is not a new number: it is the ceiling this driver has always advertised
#: — `MAX_LEASE_S` in the actuator, the cap `test_lease_is_capped` asserts, and
#: the value receipts have been reporting all along. What was missing was the
#: enforcement, so the ceiling moves HERE, into the thread that actually stops
#: the car, where a caller cannot route around it. At the driver's 0.35 throttle
#: ceiling a 2 s lease is a couple of metres of unattended travel; that is the
#: most a single un-renewed command may ever buy, and it is why the honest
#: ceiling still belongs well under a human reaction time's worth of distance.
MAX_LEASE_S = 2.0

#: How often the watchdog checks, and how often it re-asserts neutral once
#: expired. Must be well under the timeout or expiry is detected late.
#:
#: This is also the honest error bar on any promised stop time: expiry is
#: DETECTED on a tick, so the wheels stop somewhere in [lease, lease + TICK_S].
#: Receipts carry this number rather than implying a precision that a polling
#: watchdog on Linux does not have.
TICK_S = 0.05


class Deadman:
    """Holds the vehicle live only while commands keep arriving."""

    def __init__(
        self,
        stop: Callable[[], None],
        timeout_s: float = DEFAULT_TIMEOUT_S,
        tick_s: float = TICK_S,
        max_lease_s: float = MAX_LEASE_S,
    ) -> None:
        """
        Args:
            stop: Brings the vehicle to a safe state. MUST be idempotent and
                safe to call at any time, including repeatedly — it is invoked
                on a timer, not as a one-shot.
            timeout_s: Lease granted to a feed that states no duration. The
                short, conservative default.
            max_lease_s: Hard ceiling on any single feed, however long the
                duration it asks for. May only TIGHTEN `MAX_LEASE_S`.
        """
        self._stop = stop
        self._timeout_s = max(0.0, float(timeout_s))
        self._tick_s = tick_s
        # Clamped to MAX_LEASE_S rather than trusted: the ceiling may be
        # TIGHTENED by whoever builds the watchdog and never loosened, so a
        # misconfiguration — or a caller that got hold of the constructor —
        # cannot buy a longer lease than the design allows.
        self._max_lease_s = min(max(0.0, float(max_lease_s)), MAX_LEASE_S)
        # Same rule for the no-duration default: it is a floor of safety, not a
        # way around the ceiling.
        self._timeout_s = min(self._timeout_s, self._max_lease_s)
        self._lock = threading.Lock()
        #: The lease the most recent feed actually granted, and the instant it
        #: runs out. `_expires_at` is the SINGLE source of truth for when the
        #: wheels stop — telemetry, receipts and the watchdog thread all read
        #: it, so there is no second number that can drift away from the first.
        #: (There used to be: a last-fed timestamp here and a duration in the
        #: receipt, which is how they came to disagree by 5x.)
        #:
        #: Deliberately starts EXPIRED: before any command exists, neutral is
        #: the only defensible state.
        self._lease_s = 0.0
        self._expires_at = 0.0
        self._expired = True
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="rc-car-deadman")
        self._thread.start()

    def clamp_lease(self, for_s: float | None = None) -> float:
        """The lease a feed WOULD grant. Pure — grants nothing, changes nothing.

        Exists so a caller can find out what it is about to get before it
        commits to a receipt: the bound is applied in one place and read from
        that same place, rather than being reimplemented by whoever writes the
        signed record.
        """
        if for_s is None:
            return min(self._timeout_s, self._max_lease_s)
        value = float(for_s)
        if value != value:  # NaN
            # Same rule as a NaN throttle: a duration nobody can interpret
            # becomes the SAFE value, and for a lease the safe value is none at
            # all. NaN silently defeats every comparison below, so it is caught
            # here rather than left to survive the clamps as an unbounded lease.
            return 0.0
        return max(0.0, min(value, self._max_lease_s))

    def feed(self, for_s: float | None = None) -> float:
        """Renew the vehicle's licence to move. Call ONLY on actuation.

        Args:
            for_s: How long this feed should keep the car alive. Clamped to
                `max_lease_s`; None means the short default.

        Returns:
            The lease ACTUALLY granted, in seconds. Report this number, never
            the requested one — the whole point is that the record and the
            mechanism agree.
        """
        granted = self.clamp_lease(for_s)
        if granted <= 0.0:
            # A zero-length lease is a stop, not a renewal. Expiring here rather
            # than arming for zero seconds means the car is neutral NOW instead
            # of at the next tick.
            self.expire_now()
            return 0.0
        with self._lock:
            now = time.monotonic()
            self._lease_s = granted
            self._expires_at = now + granted
            if self._expired:
                logger.info("deadman re-armed for %.3fs", granted)
            self._expired = False
        return granted

    def expire_now(self) -> None:
        """Stop immediately — the operator asked, or an envelope ran out.

        Outranks any live lease, however long, and takes no lock the command
        path holds.
        """
        with self._lock:
            self._lease_s = 0.0
            self._expires_at = 0.0
            self._expired = True
        self._safe_stop()

    @property
    def alive(self) -> bool:
        with self._lock:
            return not self._expired

    @property
    def lease_s(self) -> float:
        """The lease currently in force, or 0.0 when expired."""
        with self._lock:
            return 0.0 if self._expired else self._lease_s

    @property
    def expires_at_monotonic(self) -> float | None:
        """The monotonic instant the current lease runs out; None when expired.

        This is the number a receipt must carry. Expiry is detected on a tick,
        so the wheels actually go neutral within `tick_s` AFTER this instant —
        never before it.
        """
        with self._lock:
            return None if self._expired else self._expires_at

    @property
    def tick_s(self) -> float:
        """Detection granularity, i.e. the error bar on the promised stop time."""
        return self._tick_s

    @property
    def max_lease_s(self) -> float:
        """The longest lease this watchdog will grant to any single feed."""
        return self._max_lease_s

    @property
    def seconds_remaining(self) -> float:
        with self._lock:
            if self._expired:
                return 0.0
            return max(0.0, self._expires_at - time.monotonic())

    def shutdown(self) -> None:
        """Stop the watchdog — and the vehicle. Never leave it moving."""
        self._running = False
        self.expire_now()

    # -- internals ---------------------------------------------------------

    def _safe_stop(self) -> None:
        try:
            self._stop()
        except Exception:
            # A raising stop() must not kill the watchdog thread; if it did,
            # nothing would ever try to stop the vehicle again. Log and retry
            # on the next tick.
            logger.exception("deadman stop() raised; will retry next tick")

    def _run(self) -> None:
        while self._running:
            time.sleep(self._tick_s)
            with self._lock:
                if self._expired:
                    just_expired = False
                    lease = 0.0
                else:
                    lease = self._lease_s
                    just_expired = time.monotonic() >= self._expires_at
                    if just_expired:
                        self._expired = True
                expired_now = self._expired
            if just_expired:
                logger.warning("deadman EXPIRED after its %.3fs lease — stopping",
                               lease)
            if expired_now:
                # Re-assert every tick while expired. One write can be lost;
                # "we asked it to stop once" is not a safety guarantee.
                self._safe_stop()
