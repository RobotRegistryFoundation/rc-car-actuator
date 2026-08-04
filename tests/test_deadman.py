"""The deadman is the safety story for a moving vehicle, so it is tested first
and hardest. Every test here describes a real way a car keeps driving when it
should not.

The lease tests at the bottom measure WHEN THE CAR ACTUALLY STOPS rather than
what the deadman was asked for. That distinction is not pedantry: this class
once ignored the requested duration entirely while the actuator signed receipts
quoting it, and every test in the package passed, because they all asserted the
reported number and none of them timed the vehicle.
"""
import threading
import time

import pytest

from rc_car_actuator.deadman import DEFAULT_TIMEOUT_S, MAX_LEASE_S, Deadman


class StopRecorder:
    def __init__(self, raises: bool = False):
        self.calls = 0
        self.raises = raises
        self._lock = threading.Lock()

    def __call__(self):
        with self._lock:
            self.calls += 1
        if self.raises:
            raise RuntimeError("PWM bus glitch")


def test_starts_expired_so_boot_state_is_neutral():
    """Before any command exists, the only defensible state is stopped."""
    stop = StopRecorder()
    dm = Deadman(stop, timeout_s=0.2, tick_s=0.02)
    try:
        assert not dm.alive
        time.sleep(0.1)
        # It should already be asserting neutral without anyone asking.
        assert stop.calls > 0
    finally:
        dm.shutdown()


def test_feeding_keeps_it_alive():
    stop = StopRecorder()
    dm = Deadman(stop, timeout_s=0.3, tick_s=0.02)
    try:
        dm.feed()
        assert dm.alive
        for _ in range(6):
            time.sleep(0.05)
            dm.feed()
        assert dm.alive, "steady feeding must not expire"
    finally:
        dm.shutdown()


def test_expires_when_commands_stop():
    """The core case: the phone dies mid-drive."""
    stop = StopRecorder()
    dm = Deadman(stop, timeout_s=0.15, tick_s=0.02)
    try:
        dm.feed()
        assert dm.alive
        before = stop.calls
        time.sleep(0.35)
        assert not dm.alive
        assert stop.calls > before, "must actually stop, not merely flag expiry"
    finally:
        dm.shutdown()


def test_keeps_reasserting_neutral_while_expired():
    """One write can be lost to a bus glitch. Once is not a guarantee."""
    stop = StopRecorder()
    dm = Deadman(stop, timeout_s=0.1, tick_s=0.02)
    try:
        dm.feed()
        time.sleep(0.2)
        first = stop.calls
        time.sleep(0.2)
        assert stop.calls > first, "must re-assert neutral, not stop once"
    finally:
        dm.shutdown()


def test_a_raising_stop_does_not_kill_the_watchdog():
    """If stop() throws and the thread dies, nothing ever stops the car again."""
    stop = StopRecorder(raises=True)
    dm = Deadman(stop, timeout_s=0.1, tick_s=0.02)
    try:
        dm.feed()
        time.sleep(0.3)
        first = stop.calls
        time.sleep(0.2)
        assert stop.calls > first, "watchdog must survive a raising stop()"
    finally:
        dm.shutdown()


def test_expire_now_stops_immediately():
    stop = StopRecorder()
    dm = Deadman(stop, timeout_s=10.0, tick_s=0.02)
    try:
        dm.feed()
        assert dm.alive
        dm.expire_now()
        assert not dm.alive
        assert stop.calls > 0
    finally:
        dm.shutdown()


def test_shutdown_leaves_the_vehicle_stopped():
    stop = StopRecorder()
    dm = Deadman(stop, timeout_s=10.0, tick_s=0.02)
    dm.feed()
    dm.shutdown()
    assert not dm.alive
    assert stop.calls > 0, "shutting down must never leave it moving"


def test_seconds_remaining_reports_the_budget():
    stop = StopRecorder()
    dm = Deadman(stop, timeout_s=0.5, tick_s=0.02)
    try:
        dm.feed()
        remaining = dm.seconds_remaining
        assert 0 < remaining <= 0.5
        dm.expire_now()
        assert dm.seconds_remaining == 0.0
    finally:
        dm.shutdown()


# --------------------------------------------------------------------------- #
# The lease is per-feed, and the number granted is the number enforced
# --------------------------------------------------------------------------- #

def time_to_expiry(dm, limit_s: float = 5.0) -> float:
    """Seconds until the watchdog actually declares the lease dead.

    Polls the mechanism instead of trusting arithmetic about it — this is the
    measurement that was missing when the driver shipped a 5x-wrong receipt.
    """
    start = time.monotonic()
    while dm.alive:
        if time.monotonic() - start > limit_s:
            return float("inf")
        time.sleep(0.005)
    return time.monotonic() - start


def test_a_feed_lasts_as_long_as_it_asked_for():
    """The defect, stated as a test: a 0.6 s feed must buy 0.6 s of motion.

    Before the fix this expired at the constructor's fixed 0.15 s while the
    receipt above it claimed the full request.
    """
    stop = StopRecorder()
    dm = Deadman(stop, timeout_s=0.15, tick_s=0.02)
    try:
        granted = dm.feed(for_s=0.6)
        assert granted == pytest.approx(0.6)
        measured = time_to_expiry(dm)
        assert measured == pytest.approx(0.6, abs=0.12), (
            f"asked for 0.6 s, granted {granted} s, actually stopped after "
            f"{measured:.3f} s")
        assert stop.calls > 0, "expiry must actually stop the vehicle"
    finally:
        dm.shutdown()


def test_a_feed_with_no_duration_takes_the_short_default():
    """Silence about duration must buy the conservative default, not the ceiling."""
    stop = StopRecorder()
    dm = Deadman(stop, timeout_s=0.2, tick_s=0.02)
    try:
        assert dm.feed() == pytest.approx(0.2)
        measured = time_to_expiry(dm)
        assert measured == pytest.approx(0.2, abs=0.12)
    finally:
        dm.shutdown()

    assert Deadman.__init__.__defaults__[0] == DEFAULT_TIMEOUT_S, \
        "the no-duration default must stay the short one"


def test_a_shorter_lease_than_the_default_is_honoured():
    """Asking for less is always allowed. Every bound here only ever cuts."""
    stop = StopRecorder()
    dm = Deadman(stop, timeout_s=1.0, tick_s=0.02)
    try:
        assert dm.feed(for_s=0.15) == pytest.approx(0.15)
        measured = time_to_expiry(dm)
        assert measured == pytest.approx(0.15, abs=0.1), \
            f"a 0.15 s request ran for {measured:.3f} s"
    finally:
        dm.shutdown()


def test_a_long_request_is_clamped_to_the_ceiling():
    """A lease is not a schedule. Thirty seconds of throttle is not on offer."""
    stop = StopRecorder()
    dm = Deadman(stop, timeout_s=0.1, tick_s=0.02, max_lease_s=0.3)
    try:
        assert dm.feed(for_s=30.0) == pytest.approx(0.3)
        measured = time_to_expiry(dm)
        assert measured == pytest.approx(0.3, abs=0.12), \
            f"a clamped 0.3 s lease ran for {measured:.3f} s"
    finally:
        dm.shutdown()


def test_the_ceiling_cannot_be_widened_by_whoever_builds_the_watchdog():
    """`max_lease_s` may tighten MAX_LEASE_S and never loosen it."""
    stop = StopRecorder()
    dm = Deadman(stop, timeout_s=0.1, tick_s=0.02, max_lease_s=600.0)
    try:
        assert dm.max_lease_s == MAX_LEASE_S
        assert dm.clamp_lease(600.0) == MAX_LEASE_S
    finally:
        dm.shutdown()


def test_a_default_longer_than_the_ceiling_is_cut_to_it():
    stop = StopRecorder()
    dm = Deadman(stop, timeout_s=10.0, tick_s=0.02, max_lease_s=0.2)
    try:
        assert dm.feed() == pytest.approx(0.2)
    finally:
        dm.shutdown()


def test_a_nan_duration_is_a_stop_not_an_unbounded_lease():
    """NaN defeats every comparison silently; it must land on the safe value."""
    stop = StopRecorder()
    dm = Deadman(stop, timeout_s=0.5, tick_s=0.02)
    try:
        dm.feed(for_s=1.0)
        assert dm.alive
        assert dm.feed(for_s=float("nan")) == 0.0
        assert not dm.alive, "a NaN duration left the car holding a live lease"
    finally:
        dm.shutdown()


def test_a_zero_lease_is_a_stop():
    stop = StopRecorder()
    dm = Deadman(stop, timeout_s=0.5, tick_s=0.02)
    try:
        dm.feed(for_s=1.0)
        before = stop.calls
        assert dm.feed(for_s=0.0) == 0.0
        assert not dm.alive
        assert stop.calls > before, "a zero lease must stop the car, not idle it"
    finally:
        dm.shutdown()


def test_expire_now_outranks_a_live_lease_however_long():
    """A stop that a long lease can outlive is not a stop."""
    stop = StopRecorder()
    dm = Deadman(stop, timeout_s=0.1, tick_s=0.02)
    try:
        assert dm.feed(for_s=MAX_LEASE_S) == pytest.approx(MAX_LEASE_S)
        before = stop.calls
        dm.expire_now()
        assert not dm.alive
        assert dm.seconds_remaining == 0.0
        assert dm.lease_s == 0.0
        assert stop.calls > before
    finally:
        dm.shutdown()


def test_the_reported_expiry_is_the_enforced_expiry():
    """`expires_at_monotonic` is what a receipt quotes, so it must be the truth.

    The wheels may stop within one tick AFTER the promised instant — detection
    is a poll — but never before it, and never a lease later.
    """
    stop = StopRecorder()
    dm = Deadman(stop, timeout_s=0.1, tick_s=0.02)
    try:
        granted = dm.feed(for_s=0.4)
        promised = dm.expires_at_monotonic
        assert promised is not None
        assert promised - time.monotonic() == pytest.approx(granted, abs=0.05)

        time_to_expiry(dm)
        overshoot = time.monotonic() - promised
        assert overshoot >= 0.0, "the car stopped BEFORE the receipt said it would"
        assert overshoot <= dm.tick_s + 0.05, \
            f"the car stopped {overshoot:.3f}s after the promised instant"
        assert dm.expires_at_monotonic is None
    finally:
        dm.shutdown()


def test_a_renewed_lease_replaces_the_old_one_rather_than_extending_it():
    """Feeding for less must SHORTEN the lease, not leave the longer one running."""
    stop = StopRecorder()
    dm = Deadman(stop, timeout_s=0.1, tick_s=0.02)
    try:
        dm.feed(for_s=1.5)
        dm.feed(for_s=0.2)
        assert dm.lease_s == pytest.approx(0.2)
        measured = time_to_expiry(dm)
        assert measured == pytest.approx(0.2, abs=0.1), \
            f"a re-fed 0.2 s lease ran for {measured:.3f} s"
    finally:
        dm.shutdown()
