"""Choosing a drive backend, and the two ways that choice can go wrong.

Every test here is about a default. The dangerous direction is getting hardware
you did not ask for; the expensive direction is asking for hardware and silently
getting a simulator, which looks exactly like correct software driving a broken
car.
"""
from __future__ import annotations

import pytest

from rc_car_actuator.backend import (
    ENV_BACKEND,
    DriveConfigError,
    drive_from_env,
)
from rc_car_actuator.drive import SimulatedDrive


def test_an_empty_environment_gets_the_simulator():
    assert isinstance(drive_from_env({}), SimulatedDrive)


@pytest.mark.parametrize("value", ["simulated", "sim", "none", "SIMULATED", " sim "])
def test_the_simulator_can_be_asked_for_by_name(value):
    assert isinstance(drive_from_env({ENV_BACKEND: value}), SimulatedDrive)


def test_an_unknown_backend_is_an_error_not_a_fallback():
    # A typo'd `pca8695` must not quietly become simulation: the wheels would
    # sit still while every receipt reported a granted lease, and the wiring
    # would get torn apart looking for a fault that is a spelling mistake.
    with pytest.raises(DriveConfigError) as exc:
        drive_from_env({ENV_BACKEND: "pca8695"})
    assert "pca8695" in str(exc.value)


def test_asking_for_real_hardware_that_is_unavailable_raises(monkeypatch):
    # The whole point of the module. On a machine with no smbus2 and no I2C bus,
    # OPENCASTOR_DRIVE=pca9685 must fail loudly rather than hand back something
    # that cannot move.
    import builtins

    real_import = builtins.__import__

    def no_smbus(name, *args, **kwargs):
        if name == "smbus2":
            raise ImportError("no smbus2 here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_smbus)
    with pytest.raises(DriveConfigError) as exc:
        drive_from_env({ENV_BACKEND: "pca9685"})
    assert "smbus2" in str(exc.value)


def test_a_malformed_number_is_reported_against_its_own_variable():
    with pytest.raises(DriveConfigError) as exc:
        drive_from_env({ENV_BACKEND: "pca9685",
                        "OPENCASTOR_DRIVE_THROTTLE_NEUTRAL_US": "about 1500"})
    assert "OPENCASTOR_DRIVE_THROTTLE_NEUTRAL_US" in str(exc.value)


def test_the_i2c_address_is_read_in_the_notation_everyone_writes_it_in():
    # Every datasheet, breakout silkscreen and i2cdetect line says 0x40.
    from rc_car_actuator.backend import _int

    assert _int({}, "MISSING", 0x40) == 0x40
    assert _int({"ADDR": "0x41"}, "ADDR", 0x40) == 0x41
    assert _int({"ADDR": "65"}, "ADDR", 0x40) == 65


def test_the_readers_read_the_environment_they_were_handed():
    # Pinned because the first version of backend.py took an `environ` argument
    # at the top and then read os.environ three functions down, which made every
    # validation test silently exercise the defaults.
    from rc_car_actuator.backend import _bool, _float, _int

    assert _float({"A": "2.5"}, "A", 1.0) == 2.5
    assert _int({"A": "7"}, "A", 1) == 7
    assert _bool({"A": "yes"}, "A") is True
    assert _bool({}, "A") is False


def test_the_actuator_defaults_to_simulation_with_no_arguments(monkeypatch):
    # The gateway constructs actuators as `cls()`. That path must never reach
    # real hardware by accident.
    monkeypatch.delenv(ENV_BACKEND, raising=False)
    from rc_car_actuator.actuator import RCCarActuator

    actuator = RCCarActuator()
    assert actuator.read_state()["hardware"] == "SimulatedDrive"


def test_the_actuator_honours_an_explicitly_passed_backend():
    from rc_car_actuator.actuator import RCCarActuator

    hardware = SimulatedDrive()
    actuator = RCCarActuator(hardware=hardware)
    actuator.envelope_open(motion_budget_s=5, window_s=30, max_throttle=0.2,
                           approved_by="test")
    actuator.drive_set(throttle=0.1, steering=0.0, duration_s=0.5)
    assert hardware.last != (0.0, 0.0), "the passed-in hardware is the one driven"
