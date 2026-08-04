"""Tests for the drive actuator.

These run against `SimulatedDrive`, which records commands and moves nothing, so
the safety behaviour is genuinely exercised rather than asserted about code
nobody ran.

The most important test in this file is
`test_execute_returns_immediately_and_does_not_sleep_for_the_duration`. If that
one ever starts passing for the wrong reason — because someone "helpfully" made
execute() wait for the motion to finish — the stop command would queue behind
the motion it exists to cancel.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from rc_car_actuator.actuator import MAX_LEASE_S, RCCarActuator
from rc_car_actuator.drive import SimulatedDrive

MANIFEST = Path("/tmp/does-not-matter.md")


@pytest.fixture
def car():
    """A car with a generous approval already open.

    Most tests here are about the lease and the hardware, not about
    authorisation, so they start from an approved state. The tests that care
    about approval build their own actuator — see the envelope section.
    """
    hw = SimulatedDrive()
    actuator = RCCarActuator(hardware=hw, lease_timeout_s=0.3)
    actuator.envelope_open(motion_budget_s=60.0, window_s=300.0,
                           max_throttle=1.0, approved_by="test")
    yield actuator, hw
    actuator.shutdown()


@pytest.fixture
def unapproved_car():
    """A car with no approval — the default state of a freshly built driver."""
    hw = SimulatedDrive()
    actuator = RCCarActuator(hardware=hw, lease_timeout_s=0.3)
    yield actuator, hw
    actuator.shutdown()


def invoke(actuator, tool, args=None, tier="actuate"):
    return actuator.execute(
        envelope={"tool_name": tool, "tool_args": args or {}, "scope": "CONTROL"},
        manifest_path=MANIFEST,
        tier=tier,
        config={},
    )


# --------------------------------------------------------------------------- #
# The lease, and the fact that execute() must not block
# --------------------------------------------------------------------------- #

def test_execute_returns_immediately_and_does_not_sleep_for_the_duration(car):
    """A 2 s lease must not take 2 s to command.

    This is the whole design. A blocking execute() would hold the request path
    for the duration of the motion, so the next request — "stop" — would arrive
    only after the motion it was meant to cancel had already finished.
    """
    actuator, _ = car
    start = time.monotonic()
    outcome = invoke(actuator, "drive.set",
                     {"throttle": 0.2, "steering": 0.0, "duration_s": 2.0})
    elapsed = time.monotonic() - start

    assert outcome.success is True
    assert elapsed < 0.1, f"execute() blocked for {elapsed:.3f}s — it must return at once"
    assert outcome.telemetry["blocking"] is False
    assert outcome.telemetry["lease_s"] == 2.0


def test_car_stops_on_its_own_when_the_lease_expires(car):
    """Nobody sent a stop. The car must stop anyway.

    The sleep is derived from the lease the RECEIPT reported, not from a number
    written into the test. Both used to be here, they disagreed by 5x, and the
    test passed on the shorter one — which is precisely how a driver ships that
    signs receipts it does not honour.
    """
    actuator, hw = car
    outcome = invoke(actuator, "drive.set",
                     {"throttle": 0.3, "steering": 0.0, "duration_s": 0.4})
    assert hw.last[0] > 0
    lease = outcome.telemetry["lease_s"]

    # Deliberately send nothing — this is the phone-locked / Wi-Fi-dropped case.
    time.sleep(lease + 0.25)
    assert hw.last == (0.0, 0.0), "the deadman did not return the car to neutral"
    assert actuator.read_state()["moving"] is False


def test_neutral_is_re_asserted_repeatedly_while_expired(car):
    """One stop write is not a safety story — a single write can be lost."""
    actuator, hw = car
    outcome = invoke(actuator, "drive.set",
                     {"throttle": 0.3, "steering": 0.0, "duration_s": 0.4})
    time.sleep(outcome.telemetry["lease_s"] + 0.2)
    first = hw.neutral_count
    time.sleep(0.25)
    assert hw.neutral_count > first, "neutral was written once and then abandoned"


def test_repeated_commands_keep_the_car_alive(car):
    """The lease renews. A driver holding the stick down keeps driving."""
    actuator, hw = car
    deadline = time.monotonic() + 0.6
    while time.monotonic() < deadline:
        invoke(actuator, "drive.set", {"throttle": 0.2, "steering": 0.0, "duration_s": 0.5})
        time.sleep(0.05)
    assert hw.last[0] > 0, "the car stopped despite being fed continuously"
    assert actuator.read_state()["lease_alive"] is True


def test_lease_is_capped(car):
    """A lease is not a schedule — asking for a minute must not grant one."""
    actuator, _ = car
    outcome = invoke(actuator, "drive.set",
                     {"throttle": 0.1, "steering": 0.0, "duration_s": 60.0})
    assert outcome.telemetry["lease_s"] == MAX_LEASE_S


def test_zero_duration_is_a_stop_not_a_no_op(car):
    """Ignoring a zero-length lease would leave the previous throttle running."""
    actuator, hw = car
    invoke(actuator, "drive.set", {"throttle": 0.3, "steering": 0.0, "duration_s": 1.0})
    assert hw.last[0] > 0
    invoke(actuator, "drive.set", {"throttle": 0.3, "steering": 0.0, "duration_s": 0.0})
    assert hw.last == (0.0, 0.0)


# --------------------------------------------------------------------------- #
# The receipt must be true about when the wheels stop
#
# Everything above asserts the number the driver REPORTS. That is exactly how a
# 5x-wrong lease shipped: `lease_s: 2.0` on a signed receipt while a fixed
# 0.4 s timeout stopped the car. These tests time the vehicle instead.
# --------------------------------------------------------------------------- #

def seconds_until_stopped(actuator, limit_s: float = 6.0) -> float:
    """Seconds until telemetry says the lease is dead. Polls the mechanism."""
    start = time.monotonic()
    while actuator.read_state()["lease_alive"]:
        if time.monotonic() - start > limit_s:
            return float("inf")
        time.sleep(0.005)
    return time.monotonic() - start


def test_the_reported_lease_is_the_lease_actually_enforced(car):
    """THE defect. A receipt that says 0.6 s must buy 0.6 s of motion.

    Signed evidence is only worth the mechanism behind it. A receipt asserting a
    stop time the deadman will not honour is worse than an unsigned one, because
    it invites someone to rely on it.
    """
    actuator, hw = car
    outcome = invoke(actuator, "drive.set",
                     {"throttle": 0.3, "steering": 0.0, "duration_s": 0.6})
    reported = outcome.telemetry["lease_s"]
    assert reported == pytest.approx(0.6)
    assert hw.last[0] > 0

    measured = seconds_until_stopped(actuator)
    assert measured == pytest.approx(reported, abs=0.15), (
        f"the receipt promised {reported}s of motion and the car stopped after "
        f"{measured:.3f}s")
    assert hw.last == (0.0, 0.0)


def test_the_receipt_stop_instant_matches_the_deadman(car):
    """`stops_at_monotonic` is read back from the watchdog, not recomputed."""
    actuator, hw = car
    outcome = invoke(actuator, "drive.set",
                     {"throttle": 0.3, "steering": 0.0, "duration_s": 0.5})
    promised = outcome.telemetry["stops_at_monotonic"]

    seconds_until_stopped(actuator)
    overshoot = time.monotonic() - promised
    assert overshoot >= 0.0, "the car stopped BEFORE the receipt said it would"
    assert overshoot <= outcome.telemetry["stop_detect_within_s"] + 0.1, (
        f"the car stopped {overshoot:.3f}s after the instant the receipt named")


def test_a_capped_lease_reports_and_enforces_the_cap():
    """Asking for a minute reports the ceiling — and stops at the ceiling."""
    hw = SimulatedDrive()
    # The ceiling may be tightened for a test the same way an operator would
    # tighten it for a tow-test; what it may never do is widen.
    actuator = RCCarActuator(hardware=hw, lease_timeout_s=0.2, max_lease_s=0.5)
    actuator.envelope_open(motion_budget_s=60.0, window_s=300.0, max_throttle=1.0)
    try:
        outcome = invoke(actuator, "drive.set",
                         {"throttle": 0.3, "steering": 0.0, "duration_s": 60.0})
        assert outcome.telemetry["lease_s"] == pytest.approx(0.5)
        assert outcome.telemetry["requested_lease_s"] == 60.0
        assert outcome.telemetry["lease_cut_by"] == "ceiling"

        measured = seconds_until_stopped(actuator)
        assert measured == pytest.approx(0.5, abs=0.15), \
            f"a lease clamped to 0.5s ran for {measured:.3f}s"
        assert hw.last == (0.0, 0.0)
    finally:
        actuator.shutdown()


def test_the_ceiling_cannot_be_widened_by_construction():
    """An operator may tighten the per-command ceiling and never loosen it."""
    hw = SimulatedDrive()
    actuator = RCCarActuator(hardware=hw, max_lease_s=600.0)
    try:
        assert actuator.read_state()["lease_max_s"] == MAX_LEASE_S
    finally:
        actuator.shutdown()


def test_a_command_with_no_duration_takes_the_short_default(car):
    """No opinion about duration buys the conservative default, not the ceiling."""
    actuator, _ = car
    telemetry = actuator.drive_set(throttle=0.2, steering=0.0)
    assert telemetry["lease_s"] == pytest.approx(0.3)   # the fixture's default
    assert telemetry["requested_lease_s"] is None
    measured = seconds_until_stopped(actuator)
    assert measured == pytest.approx(0.3, abs=0.15)


def test_stop_kills_a_long_live_lease_immediately(car):
    """A stop must not have to wait out the lease it is cancelling."""
    actuator, hw = car
    outcome = invoke(actuator, "drive.set",
                     {"throttle": 0.3, "steering": 0.0, "duration_s": MAX_LEASE_S})
    assert outcome.telemetry["lease_s"] == pytest.approx(MAX_LEASE_S)
    assert hw.last[0] > 0

    start = time.monotonic()
    invoke(actuator, "drive.stop", tier="read")
    elapsed = time.monotonic() - start
    assert hw.last == (0.0, 0.0), "a 2 s lease outlived a stop"
    assert actuator.read_state()["lease_alive"] is False
    assert elapsed < 0.1, f"the stop took {elapsed:.3f}s — it must be immediate"


def test_revoke_kills_a_long_live_lease_immediately(car):
    """Withdrawing the approval must stop the car, not wait for its lease."""
    actuator, hw = car
    invoke(actuator, "drive.set",
           {"throttle": 0.3, "steering": 0.0, "duration_s": MAX_LEASE_S})
    assert hw.last[0] > 0

    invoke(actuator, "drive.envelope.revoke", tier="read")
    assert hw.last == (0.0, 0.0), "a 2 s lease outlived a revocation"
    assert actuator.read_state()["lease_alive"] is False


def test_estop_kills_a_long_live_lease_immediately(car):
    actuator, hw = car
    invoke(actuator, "drive.set",
           {"throttle": 0.3, "steering": 0.0, "duration_s": MAX_LEASE_S})
    actuator.estop()
    assert hw.last == (0.0, 0.0), "a 2 s lease outlived an e-stop"
    assert actuator.read_state()["lease_alive"] is False


# --------------------------------------------------------------------------- #
# Starting state
# --------------------------------------------------------------------------- #

def test_starts_stopped_and_not_alive():
    """Before any command exists, neutral is the only defensible state."""
    hw = SimulatedDrive()
    actuator = RCCarActuator(hardware=hw)
    try:
        assert hw.last == (0.0, 0.0)
        assert actuator.read_state()["lease_alive"] is False
        assert actuator.read_state()["moving"] is False
    finally:
        actuator.shutdown()


def test_default_hardware_cannot_move_a_real_vehicle():
    """A driver built with no arguments must be inert."""
    actuator = RCCarActuator()
    try:
        assert actuator.read_state()["hardware"] == "SimulatedDrive"
    finally:
        actuator.shutdown()


# --------------------------------------------------------------------------- #
# Bounds
# --------------------------------------------------------------------------- #

def test_throttle_is_capped_and_the_cap_is_reported(car):
    actuator, hw = car
    actuator.envelope_open(motion_budget_s=60.0, window_s=300.0, max_throttle=1.0)
    outcome = invoke(actuator, "drive.set",
                     {"throttle": 1.0, "steering": 0.0, "duration_s": 0.5})
    assert hw.last[0] == pytest.approx(0.35)
    assert outcome.telemetry["throttle_capped"] is True
    assert outcome.telemetry["requested_throttle"] == 1.0


def test_reverse_is_capped_too(car):
    actuator, hw = car
    actuator.envelope_open(motion_budget_s=60.0, window_s=300.0, max_throttle=1.0)
    invoke(actuator, "drive.set", {"throttle": -1.0, "steering": 0.0, "duration_s": 0.5})
    assert hw.last[0] == pytest.approx(-0.35)


def test_nan_throttle_becomes_zero(car):
    """NaN fails every comparison silently and would reach PWM as undefined."""
    actuator, hw = car
    invoke(actuator, "drive.set",
           {"throttle": float("nan"), "steering": float("nan"), "duration_s": 0.5})
    assert hw.last == (0.0, 0.0)


def test_steering_is_bounded_but_not_throttle_capped(car):
    """Steering uses full scale; only throttle carries the speed ceiling."""
    actuator, hw = car
    invoke(actuator, "drive.set", {"throttle": 0.0, "steering": 5.0, "duration_s": 0.5})
    assert hw.last[1] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Stop and e-stop
# --------------------------------------------------------------------------- #

def test_stop_is_available_to_a_read_tier_caller(car):
    """A stop that can be refused is not a stop."""
    actuator, hw = car
    invoke(actuator, "drive.set", {"throttle": 0.3, "steering": 0.0, "duration_s": 1.0})
    outcome = invoke(actuator, "drive.stop", tier="read")
    assert outcome.success is True
    assert hw.last == (0.0, 0.0)


def test_read_tier_cannot_command_motion(car):
    actuator, hw = car
    outcome = invoke(actuator, "drive.set",
                     {"throttle": 0.3, "steering": 0.0, "duration_s": 1.0},
                     tier="read")
    assert outcome.success is False
    assert outcome.outcome_kind == "denied"
    assert hw.last == (0.0, 0.0)


def test_anon_tier_can_do_nothing(car):
    actuator, _ = car
    for tool in ("drive.set", "drive.stop", "status.report"):
        outcome = invoke(actuator, tool, {"duration_s": 0.5}, tier="anon")
        assert outcome.success is False, f"{tool} accepted an anonymous caller"


def test_estop_refuses_further_motion_until_cleared(car):
    """Unlike a plain stop, the next command must not undo it."""
    actuator, hw = car
    actuator.estop()
    outcome = invoke(actuator, "drive.set",
                     {"throttle": 0.3, "steering": 0.0, "duration_s": 1.0})
    assert outcome.success is False
    assert hw.last == (0.0, 0.0)

    actuator.clear_estop()
    outcome = invoke(actuator, "drive.set",
                     {"throttle": 0.3, "steering": 0.0, "duration_s": 1.0})
    assert outcome.success is True


def test_clearing_estop_does_not_resume_motion(car):
    """Releasing the e-stop must not itself start the car."""
    actuator, hw = car
    invoke(actuator, "drive.set", {"throttle": 0.3, "steering": 0.0, "duration_s": 1.0})
    actuator.estop()
    actuator.clear_estop()
    assert hw.last == (0.0, 0.0)
    assert actuator.read_state()["lease_alive"] is False


# --------------------------------------------------------------------------- #
# Failure handling
# --------------------------------------------------------------------------- #

def test_hardware_failure_stops_the_car_and_reports_an_error(car):
    """A partially-applied command must not leave the car running on its lease."""
    actuator, hw = car

    class Exploding(SimulatedDrive):
        def set_drive(self, throttle, steering):
            raise OSError("PWM write failed")

    actuator._hw = Exploding()
    outcome = invoke(actuator, "drive.set",
                     {"throttle": 0.3, "steering": 0.0, "duration_s": 1.0})
    assert outcome.success is False
    assert "OSError" in outcome.error_message
    assert actuator.read_state()["lease_alive"] is False


def test_unknown_capability_is_an_error_not_a_crash(car):
    actuator, _ = car
    outcome = invoke(actuator, "drive.teleport")
    assert outcome.success is False
    assert "unknown capability" in outcome.error_message


# --------------------------------------------------------------------------- #
# Telemetry honesty
# --------------------------------------------------------------------------- #

def test_state_admits_there_is_no_firmware_lease(car):
    """The caveat belongs in telemetry, not only in the source."""
    actuator, _ = car
    state = actuator.read_state()
    assert state["firmware_lease"] is False
    assert state["stop_layers"] == ["software deadman (this process)"]


def test_estop_during_a_command_wins(car):
    """An e-stop raised while a command is mid-flight must not be overridden.

    Reproduces a real interleaving: drive_set() reads `_estopped` as False, and
    the e-stop fires before the command finishes writing and feeding the lease.
    Without mutual exclusion the command's throttle lands AFTER the e-stop's
    neutral and the lease is re-armed — the car drives away from a pressed
    emergency stop, which is the worst failure this file could have.
    """
    actuator, _ = car

    class EstopMidWrite(SimulatedDrive):
        def __init__(self, outer):
            super().__init__()
            self._outer = outer
            self._fired = False

        def set_drive(self, throttle, steering):
            if not self._fired:
                self._fired = True
                self._outer.estop()   # arrives mid-command
            super().set_drive(throttle, steering)

    hw = EstopMidWrite(actuator)
    actuator._hw = hw

    actuator.execute(
        envelope={"tool_name": "drive.set",
                  "tool_args": {"throttle": 0.3, "steering": 0.0, "duration_s": 1.0},
                  "scope": "CONTROL"},
        manifest_path=MANIFEST, tier="actuate", config={},
    )
    assert hw.last == (0.0, 0.0), "throttle was applied after an e-stop"
    assert actuator.read_state()["lease_alive"] is False, "e-stopped car holds a live lease"


def test_a_wedged_command_cannot_prevent_the_stop():
    """A hardware layer that hangs must not keep the car driving.

    The command path holds `_DRIVE_LOCK` while writing. If the stop path also
    needed that lock, a hung write would block every stop forever and the car
    would run on its last throttle until the battery died. The deadman's stop
    therefore takes no lock at all.
    """
    import threading

    release = threading.Event()
    neutralled = threading.Event()

    class HangingDrive(SimulatedDrive):
        def set_drive(self, throttle, steering):
            super().set_drive(throttle, steering)
            release.wait(timeout=5)      # wedged mid-command, holding _DRIVE_LOCK

        def neutral(self):
            super().neutral()
            neutralled.set()

    hw = HangingDrive()
    actuator = RCCarActuator(hardware=hw, lease_timeout_s=0.2)
    actuator.envelope_open(motion_budget_s=60.0, window_s=300.0, max_throttle=1.0)
    try:
        wedged = threading.Thread(
            target=lambda: actuator.drive_set(throttle=0.3, steering=0.0, duration_s=1.0),
            daemon=True)
        wedged.start()

        # The command is stuck holding the lock. The stop must still get through.
        assert neutralled.wait(timeout=3), \
            "the stop path was blocked by a wedged command — the car would never stop"
    finally:
        release.set()
        actuator.shutdown()
