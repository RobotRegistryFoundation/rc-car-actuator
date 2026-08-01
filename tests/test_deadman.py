"""The deadman is the safety story for a moving vehicle, so it is tested first
and hardest. Every test here describes a real way a car keeps driving when it
should not.
"""
import threading
import time

import pytest

from rc_car_actuator.deadman import Deadman


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
