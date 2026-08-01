"""The deadman: a moving vehicle must stop when commands stop.

This is the single most important file in the package. Everything else decides
where the car goes; this decides that it stops when nobody is steering it.

The failure it exists to prevent is not exotic. A phone locks, an app crashes,
Wi-Fi drops behind a fridge, the gateway process hangs on a slow request — and
an ESC holds its last throttle command **forever**, because PWM peripherals keep
emitting the last value written to them with no further CPU involvement. That is
a car driving into a wall at the last speed anyone asked for.

So the rule is inverted: throttle is not something you set, it is something you
must keep renewing. Stop renewing for `timeout_s` and it returns to neutral.

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
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

logger = logging.getLogger("rc_car.deadman")

#: How long a single command keeps the vehicle alive.
#:
#: Short enough that a stall is felt almost immediately at walking pace, long
#: enough to ride out ordinary Wi-Fi jitter without stuttering. At 1 m/s a
#: 400 ms lapse is ~40 cm of unattended travel, which is the most that should
#: ever happen without a human in the loop.
DEFAULT_TIMEOUT_S = 0.4

#: How often the watchdog checks, and how often it re-asserts neutral once
#: expired. Must be well under the timeout or expiry is detected late.
TICK_S = 0.05


class Deadman:
    """Holds the vehicle live only while commands keep arriving."""

    def __init__(
        self,
        stop: Callable[[], None],
        timeout_s: float = DEFAULT_TIMEOUT_S,
        tick_s: float = TICK_S,
    ) -> None:
        """
        Args:
            stop: Brings the vehicle to a safe state. MUST be idempotent and
                safe to call at any time, including repeatedly — it is invoked
                on a timer, not as a one-shot.
            timeout_s: Quiet period after which the vehicle is stopped.
        """
        self._stop = stop
        self._timeout_s = timeout_s
        self._tick_s = tick_s
        self._lock = threading.Lock()
        # Deliberately starts EXPIRED: before any command exists, neutral is the
        # only defensible state.
        self._last_fed = 0.0
        self._expired = True
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="rc-car-deadman")
        self._thread.start()

    def feed(self) -> None:
        """Renew the vehicle's licence to move. Call ONLY on actuation."""
        with self._lock:
            self._last_fed = time.monotonic()
            if self._expired:
                logger.info("deadman re-armed")
            self._expired = False

    def expire_now(self) -> None:
        """Stop immediately — the operator asked, or an envelope ran out."""
        with self._lock:
            self._last_fed = 0.0
            self._expired = True
        self._safe_stop()

    @property
    def alive(self) -> bool:
        with self._lock:
            return not self._expired

    @property
    def seconds_remaining(self) -> float:
        with self._lock:
            if self._expired:
                return 0.0
            return max(0.0, self._timeout_s - (time.monotonic() - self._last_fed))

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
                else:
                    quiet = time.monotonic() - self._last_fed
                    just_expired = quiet > self._timeout_s
                    if just_expired:
                        self._expired = True
                expired_now = self._expired
            if just_expired:
                logger.warning("deadman EXPIRED after %.2fs of silence — stopping",
                               self._timeout_s)
            if expired_now:
                # Re-assert every tick while expired. One write can be lost;
                # "we asked it to stop once" is not a safety guarantee.
                self._safe_stop()
