"""Tests for the drive envelope — continuous motion under one approval.

The envelope exists because a phone driving a car sends commands at 20 Hz.
Approving each one is absurd; approving none means the vehicle moves on nobody's
authority. So a human approves a BUDGET once and every command draws it down.

These tests are about the budget being real: that it depletes, that it cannot be
topped up by the thing spending it, that running out actually stops the car
rather than merely refusing the next command, and that idling is free.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from rc_car_actuator.actuator import RCCarActuator
from rc_car_actuator.drive import SimulatedDrive
from rc_car_actuator.envelope import (
    MAX_MOTION_BUDGET_S,
    MAX_WINDOW_S,
    DriveEnvelope,
)

MANIFEST = Path("/tmp/does-not-matter.md")


@pytest.fixture
def car():
    hw = SimulatedDrive()
    actuator = RCCarActuator(hardware=hw, lease_timeout_s=0.3)
    yield actuator, hw
    actuator.shutdown()


def invoke(actuator, tool, args=None, tier="actuate"):
    return actuator.execute(
        envelope={"tool_name": tool, "tool_args": args or {}, "scope": "CONTROL"},
        manifest_path=MANIFEST, tier=tier, config={},
    )


def drive(actuator, throttle=0.3, duration=1.0):
    return invoke(actuator, "drive.set",
                  {"throttle": throttle, "steering": 0.0, "duration_s": duration})


# --------------------------------------------------------------------------- #
# No approval, no motion
# --------------------------------------------------------------------------- #

def test_a_fresh_car_may_not_move(car):
    """The default state is no authority at all."""
    actuator, hw = car
    outcome = drive(actuator)
    assert outcome.success is False
    assert "no drive approval" in outcome.error_message
    assert hw.last == (0.0, 0.0)
    assert actuator.read_state()["may_move"] is False


def test_opening_an_envelope_permits_motion(car):
    actuator, hw = car
    actuator.envelope_open(motion_budget_s=10, window_s=60, max_throttle=0.5)
    assert drive(actuator).success is True
    assert hw.last[0] > 0


# --------------------------------------------------------------------------- #
# The budget is real
# --------------------------------------------------------------------------- #

def test_motion_spends_the_budget(car):
    actuator, _ = car
    actuator.envelope_open(motion_budget_s=10, window_s=60, max_throttle=0.5)
    drive(actuator, duration=1.0)
    time.sleep(0.25)
    drive(actuator, duration=1.0)
    spent = actuator.read_state()["envelope"]["motion_spent_s"]
    assert spent >= 0.2, f"driving for 0.25s charged only {spent}s"


def test_idling_is_free(car):
    """A parked car with an open approval must not burn its budget."""
    actuator, _ = car
    actuator.envelope_open(motion_budget_s=10, window_s=60, max_throttle=0.5)
    time.sleep(0.4)
    assert actuator.read_state()["envelope"]["motion_spent_s"] == pytest.approx(0.0, abs=0.01)


def test_steering_without_throttle_is_free(car):
    """Turning the wheels of a stationary car gets it nowhere and costs nothing."""
    actuator, _ = car
    actuator.envelope_open(motion_budget_s=10, window_s=60, max_throttle=0.5)
    invoke(actuator, "drive.set", {"throttle": 0.0, "steering": 1.0, "duration_s": 1.0})
    time.sleep(0.25)
    invoke(actuator, "drive.set", {"throttle": 0.0, "steering": -1.0, "duration_s": 1.0})
    assert actuator.read_state()["envelope"]["motion_spent_s"] == pytest.approx(0.0, abs=0.02)


def test_an_abandoned_command_is_still_charged(car):
    """The deadman stops the car with nobody calling anything — bill it anyway.

    Charging only on explicit stops would make every abandoned command free, and
    abandonment is precisely what the deadman exists to handle.
    """
    actuator, _ = car
    actuator.envelope_open(motion_budget_s=10, window_s=60, max_throttle=0.5)
    drive(actuator, duration=1.0)
    time.sleep(0.3 + 0.2)          # lease expires; nobody sends anything
    spent = actuator.read_state()["envelope"]["motion_spent_s"]
    assert spent >= 0.25, f"abandoned motion was charged only {spent}s"


def test_running_out_stops_the_car_not_just_the_next_command(car):
    """Exhaustion must halt the vehicle, not merely decline the next request."""
    actuator, hw = car
    actuator.envelope_open(motion_budget_s=0.2, window_s=60, max_throttle=0.5)
    drive(actuator, duration=2.0)
    assert hw.last[0] > 0

    # The lease was capped to the remaining budget, so the car stops on its own.
    time.sleep(0.45)
    assert hw.last == (0.0, 0.0), "the car kept driving past its approved budget"


def test_lease_cannot_outlive_the_budget(car):
    """A 2 s lease with 0.3 s of budget must be cut to 0.3 s — and stop there.

    The cut is reported AND enforced: the receipt names the budget as the bound
    that did the cutting, and the car is timed to confirm it stopped when the
    receipt said rather than when the request asked.
    """
    actuator, hw = car
    actuator.envelope_open(motion_budget_s=0.3, window_s=60, max_throttle=0.5)
    outcome = drive(actuator, duration=2.0)
    granted = outcome.telemetry["lease_s"]
    assert granted <= 0.31, f"lease {granted}s exceeded the 0.3s budget"
    assert outcome.telemetry["requested_lease_s"] == 2.0
    assert outcome.telemetry["lease_cut_by"] == "budget"

    start = time.monotonic()
    while actuator.read_state()["lease_alive"] and time.monotonic() - start < 3.0:
        time.sleep(0.005)
    measured = time.monotonic() - start
    assert measured == pytest.approx(granted, abs=0.15), (
        f"the receipt promised {granted}s and the car drove for {measured:.3f}s "
        f"on a 0.3s budget")
    assert hw.last == (0.0, 0.0)


def test_the_budget_is_charged_what_the_lease_actually_enforced(car):
    """The seconds spent and the seconds granted must be the same seconds.

    Charging is by ELAPSED motion and the lease is what bounds that elapsed
    motion, so once the receipt is honest these two numbers describe the same
    physical stretch of driving. When the receipt lied they differed by 5x: a
    2 s lease on the receipt, 0.4 s of budget actually charged.
    """
    actuator, _ = car
    actuator.envelope_open(motion_budget_s=10, window_s=60, max_throttle=0.5)
    outcome = drive(actuator, duration=0.6)
    granted = outcome.telemetry["lease_s"]

    start = time.monotonic()
    while actuator.read_state()["lease_alive"] and time.monotonic() - start < 3.0:
        time.sleep(0.005)
    time.sleep(0.05)   # let the stop path close the billing segment

    spent = actuator.read_state()["envelope"]["motion_spent_s"]
    assert spent == pytest.approx(granted, abs=0.15), (
        f"the receipt granted {granted}s of motion and the budget was charged "
        f"{spent}s")
    assert spent >= granted - 0.02, \
        "the budget was charged less than the motion actually granted"


def test_exhausted_envelope_refuses_further_motion(car):
    actuator, hw = car
    actuator.envelope_open(motion_budget_s=0.2, window_s=60, max_throttle=0.5)
    drive(actuator, duration=1.0)
    time.sleep(0.5)
    outcome = drive(actuator, duration=1.0)
    assert outcome.success is False
    assert "used up" in outcome.error_message
    assert hw.last == (0.0, 0.0)


# --------------------------------------------------------------------------- #
# The budget cannot be topped up by the thing spending it
# --------------------------------------------------------------------------- #

def test_there_is_no_way_to_extend_an_envelope():
    """An envelope the spender could extend would authorise nothing."""
    env = DriveEnvelope(motion_budget_s=5, window_s=60, max_throttle=0.5)
    for forbidden in ("extend", "renew", "top_up", "add_budget", "increase"):
        assert not hasattr(env, forbidden), f"DriveEnvelope grew a {forbidden}()"


def test_budget_only_decreases():
    env = DriveEnvelope(motion_budget_s=5, window_s=60, max_throttle=0.5)
    readings = []
    for _ in range(3):
        env.begin_motion()
        time.sleep(0.1)
        env.end_motion()
        readings.append(env.remaining_s)
    assert readings == sorted(readings, reverse=True), f"budget went up: {readings}"


def test_opening_a_new_envelope_replaces_rather_than_accumulates(car):
    """Otherwise 'approve a little more' becomes 'approve without limit'."""
    actuator, _ = car
    actuator.envelope_open(motion_budget_s=5, window_s=60, max_throttle=0.5)
    first = actuator.read_state()["envelope"]["envelope_id"]
    described = actuator.envelope_open(motion_budget_s=5, window_s=60, max_throttle=0.5)
    assert described["envelope_id"] != first
    assert described["motion_budget_s"] == 5, "budgets accumulated"


def test_hard_ceilings_apply_regardless_of_request():
    """An approval flow that can grant an unbounded budget is not a bound."""
    env = DriveEnvelope(motion_budget_s=99_999, window_s=99_999, max_throttle=99)
    assert env.motion_budget_s == MAX_MOTION_BUDGET_S
    assert env.window_s == MAX_WINDOW_S
    assert env.max_throttle == 1.0


# --------------------------------------------------------------------------- #
# Expiry, revocation, ceilings
# --------------------------------------------------------------------------- #

def test_window_expires_even_when_unused(car):
    """Authorisation must decay with time, not only with use."""
    actuator, _ = car
    actuator.envelope_open(motion_budget_s=10, window_s=0.3, max_throttle=0.5)
    time.sleep(0.4)
    outcome = drive(actuator)
    assert outcome.success is False
    assert "expired" in outcome.error_message


def test_revoking_stops_the_car_immediately(car):
    actuator, hw = car
    actuator.envelope_open(motion_budget_s=10, window_s=60, max_throttle=0.5)
    drive(actuator, duration=2.0)
    assert hw.last[0] > 0
    invoke(actuator, "drive.envelope.revoke", tier="read")
    assert hw.last == (0.0, 0.0), "authority was revoked but the car kept moving"
    assert drive(actuator).success is False


def test_revoke_is_available_to_a_read_tier_caller(car):
    """Anyone who can see the car may withdraw its permission to move."""
    actuator, _ = car
    actuator.envelope_open(motion_budget_s=10, window_s=60, max_throttle=0.5)
    assert invoke(actuator, "drive.envelope.revoke", tier="read").success is True


def test_read_tier_cannot_approve_motion(car):
    """Opening an envelope IS the approval, so it needs the actuate tier."""
    actuator, _ = car
    outcome = invoke(actuator, "drive.envelope.open",
                     {"motion_budget_s": 10, "window_s": 60, "max_throttle": 0.5},
                     tier="read")
    assert outcome.success is False
    assert outcome.outcome_kind == "denied"


def test_envelope_ceiling_composes_with_the_driver_ceiling(car):
    """The lower of the two wins, in both directions."""
    actuator, hw = car
    actuator.envelope_open(motion_budget_s=10, window_s=60, max_throttle=0.1)
    drive(actuator, throttle=1.0, duration=1.0)
    assert hw.last[0] == pytest.approx(0.1), "the envelope ceiling was ignored"

    # And the driver's own cap still binds when the envelope is more generous.
    actuator.envelope_open(motion_budget_s=10, window_s=60, max_throttle=1.0)
    drive(actuator, throttle=1.0, duration=1.0)
    assert hw.last[0] == pytest.approx(0.35), "the driver ceiling was ignored"


def test_estop_blocks_approving_new_motion(car):
    actuator, _ = car
    actuator.estop()
    with pytest.raises(RuntimeError, match="e-stop"):
        actuator.envelope_open(motion_budget_s=10, window_s=60, max_throttle=0.5)


# --------------------------------------------------------------------------- #
# Attribution
# --------------------------------------------------------------------------- #

def test_telemetry_carries_the_envelope_id(car):
    """A receipt must trace back to the approval that authorised it."""
    actuator, _ = car
    described = actuator.envelope_open(motion_budget_s=10, window_s=60,
                                       max_throttle=0.5, approved_by="craig-iphone")
    state = invoke(actuator, "status.report", tier="read").telemetry
    assert state["envelope"]["envelope_id"] == described["envelope_id"]
    assert state["envelope"]["approved_by"] == "craig-iphone"


def test_unusable_reason_is_stated_in_the_approvers_terms(car):
    actuator, _ = car
    actuator.envelope_open(motion_budget_s=10, window_s=0.2, max_throttle=0.5)
    time.sleep(0.3)
    state = actuator.read_state()
    assert state["envelope"]["usable"] is False
    assert "expired" in state["envelope"]["unusable_because"]


def test_revocation_during_a_command_wins(car):
    """A revocation arriving mid-command must not be overridden by that command.

    Same shape as the e-stop race: drive_set reads the envelope as usable, the
    revocation's neutral lands, and then the command's throttle lands on top and
    re-arms the lease — a car still driving on authority that was withdrawn.
    """
    actuator, _ = car
    actuator.envelope_open(motion_budget_s=10, window_s=60, max_throttle=0.5)

    class RevokeMidWrite(SimulatedDrive):
        def __init__(self, outer):
            super().__init__()
            self._outer = outer
            self._fired = False

        def set_drive(self, throttle, steering):
            if not self._fired:
                self._fired = True
                self._outer.envelope_revoke()
            super().set_drive(throttle, steering)

    hw = RevokeMidWrite(actuator)
    actuator._hw = hw
    drive(actuator, duration=1.0)
    assert hw.last == (0.0, 0.0), "throttle was applied after the approval was revoked"
    assert actuator.read_state()["lease_alive"] is False


def test_every_motion_receipt_names_its_approval(car):
    """A journal of drive commands is useless if it cannot name the decision."""
    actuator, _ = car
    described = actuator.envelope_open(motion_budget_s=10, window_s=60, max_throttle=0.5)
    outcome = drive(actuator, duration=0.5)
    assert outcome.telemetry["envelope_id"] == described["envelope_id"]
    assert outcome.telemetry["motion_remaining_s"] <= 10.0
