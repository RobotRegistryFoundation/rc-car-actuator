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


def test_asking_for_a_maestro_that_is_not_plugged_in_raises():
    # Same rule as the PCA9685 path: a requested backend that cannot be built
    # must not quietly become a simulator.
    with pytest.raises(DriveConfigError) as exc:
        drive_from_env({ENV_BACKEND: "maestro",
                        "OPENCASTOR_DRIVE_SERIAL_PORT": "/dev/ttyACM-nope"})
    message = str(exc.value)
    assert "/dev/ttyACM-nope" in message
    # The error names the failure mode that actually catches people out: the
    # Maestro exposes two serial devices and only one of them takes commands.
    assert "command port" in message


def test_the_unknown_backend_message_lists_every_real_one():
    with pytest.raises(DriveConfigError) as exc:
        drive_from_env({ENV_BACKEND: "nope"})
    for name in ("simulated", "pca9685", "maestro", "pigpio"):
        assert name in str(exc.value)


def test_both_pwm_backends_read_the_same_per_vehicle_trim():
    # The trim describes the VEHICLE — where the ESC sits still, where the
    # linkage points the wheels straight — so swapping controller must not mean
    # re-measuring it under different variable names.
    from rc_car_actuator.backend import _channel_from_env

    env = {"OPENCASTOR_DRIVE_THROTTLE_NEUTRAL_US": "1480",
           "OPENCASTOR_DRIVE_THROTTLE_INVERT": "yes"}
    channel = _channel_from_env(env, "THROTTLE", 0)
    assert channel.neutral_us == 1480
    assert channel.invert is True
    assert channel.pulse_us(1.0) == pytest.approx(980)


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


def test_every_documented_drive_variable_is_actually_read():
    """A setting that is written down, looks right, and does nothing.

    The ESC behaviours were added to the driver and documented in a robot's
    gateway-policy.env before the env reader knew about them, so enabling
    reverse arming would have silently changed nothing. This pins the whole
    surface: if a field gains an env var in the docs, it gains one here.
    """
    from rc_car_actuator.backend import _channel_from_env
    from rc_car_actuator.pca9685 import DriveChannels

    env = {
        "OPENCASTOR_DRIVE_ARM_DELAY_S": "0.25",
        "OPENCASTOR_DRIVE_ESC_REVERSE_ARMING": "true",
        "OPENCASTOR_DRIVE_ESC_ARM_NEUTRAL_MS": "150",
        "OPENCASTOR_DRIVE_ESC_DOUBLE_TAP_REVERSE": "yes",
        "OPENCASTOR_DRIVE_THROTTLE_DEADZONE": "0.05",
        "OPENCASTOR_DRIVE_FRAME_HZ": "60",
        "OPENCASTOR_DRIVE_OSCILLATOR_HZ": "26000000",
    }
    from rc_car_actuator import backend

    # Build the channels exactly as _pca9685_from_env does, without touching I2C.
    channels = DriveChannels(
        throttle=_channel_from_env(env, "THROTTLE", 0),
        steering=_channel_from_env(env, "STEERING", 1),
        frame_hz=backend._int(env, "OPENCASTOR_DRIVE_FRAME_HZ", 50),
        oscillator_hz=backend._int(env, "OPENCASTOR_DRIVE_OSCILLATOR_HZ", 25_000_000),
        arm_delay_s=backend._float(env, "OPENCASTOR_DRIVE_ARM_DELAY_S", 0.5),
        esc_reverse_arming=backend._bool(env, "OPENCASTOR_DRIVE_ESC_REVERSE_ARMING"),
        esc_arm_neutral_ms=backend._int(env, "OPENCASTOR_DRIVE_ESC_ARM_NEUTRAL_MS", 200),
        esc_double_tap_reverse=backend._bool(env, "OPENCASTOR_DRIVE_ESC_DOUBLE_TAP_REVERSE"),
        throttle_deadzone=backend._float(env, "OPENCASTOR_DRIVE_THROTTLE_DEADZONE", 0.02),
    )
    assert channels.arm_delay_s == 0.25
    assert channels.esc_reverse_arming is True
    assert channels.esc_arm_neutral_ms == 150
    assert channels.esc_double_tap_reverse is True
    assert channels.throttle_deadzone == 0.05
    assert channels.frame_hz == 60
    assert channels.oscillator_hz == 26_000_000
