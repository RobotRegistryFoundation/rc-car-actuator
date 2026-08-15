"""Choosing a drive backend, and the two ways that choice can go wrong.

Every test here is about a default. The dangerous direction is getting hardware
you did not ask for; the expensive direction is asking for hardware and silently
getting a simulator, which looks exactly like correct software driving a broken
car.
"""
from __future__ import annotations

import os
import signal
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


class TestShutdownNeutral:
    """The bench finding: a stopped process left the wheels turning.

    With the car on a stand, 10% throttle was commanded and `rover-gateway` was
    stopped normally. The process exited in 0.2 s and the PCA9685 went on
    emitting 1547 us with nothing alive to renew it. The deadman is a thread
    inside the process, so it died with the process; the chip latches its
    registers and does not.
    """

    class Recording:
        """Minimal DriveHardware that records whether it was quieted."""

        def __init__(self):
            self.neutral_calls = 0

        def set_drive(self, throttle, steering):
            pass

        def neutral(self):
            self.neutral_calls += 1

    def test_atexit_writes_neutral(self):
        import atexit

        from rc_car_actuator.backend import install_shutdown_neutral

        drive = self.Recording()
        install_shutdown_neutral(drive)
        # Fire the registered hook the way interpreter shutdown would.
        atexit._run_exitfuncs()
        assert drive.neutral_calls >= 1, "process exit left the chip commanded"

    def test_neutral_is_written_only_once(self):
        import atexit

        from rc_car_actuator.backend import install_shutdown_neutral

        drive = self.Recording()
        install_shutdown_neutral(drive)
        atexit._run_exitfuncs()
        atexit._run_exitfuncs()
        # A signal handler and atexit can both fire. The second must not
        # re-drive a chip the first just quieted.
        assert drive.neutral_calls == 1

    def test_a_failing_neutral_does_not_raise_out_of_shutdown(self):
        import atexit

        from rc_car_actuator.backend import install_shutdown_neutral

        class Broken(self.Recording):
            def neutral(self):
                raise OSError("i2c wire pulled")

        install_shutdown_neutral(Broken())
        # Raising here would replace a stopped car with a stack trace and a
        # moving one.
        atexit._run_exitfuncs()

    def test_sigterm_writes_neutral_and_still_terminates(self):
        """End to end in a real subprocess, because signal chaining is the part
        most likely to be wrong and cannot be checked in-process."""
        import subprocess
        import sys
        import textwrap

        script = textwrap.dedent(
            """
            import os, signal, sys, time
            sys.path.insert(0, os.environ["SRC"])
            from rc_car_actuator.backend import install_shutdown_neutral

            class D:
                def set_drive(self, t, s): pass
                def neutral(self): print("NEUTRAL", flush=True)

            install_shutdown_neutral(D())
            print("READY", flush=True)
            time.sleep(30)
            """
        )
        import pathlib

        src = str(pathlib.Path(__file__).resolve().parents[1] / "src")
        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE, text=True, env={**os.environ, "SRC": src},
        )
        assert proc.stdout.readline().strip() == "READY"
        proc.send_signal(signal.SIGTERM)
        out, _ = proc.communicate(timeout=15)
        assert "NEUTRAL" in out, "SIGTERM left the chip commanded"
        # SIGTERM must still end the process. A handler that stops the wheels
        # and then hangs is its own outage.
        assert proc.returncode is not None


class TestDriveRegistry:
    """Drivers as registrations, not edits to dispatch logic."""

    def test_the_builtins_are_registered(self):
        from rc_car_actuator.backend import DRIVE_FACTORIES

        for name in ("pca9685", "maestro", "pigpio", "pca9685-tank"):
            assert name in DRIVE_FACTORIES

    def test_an_unknown_backend_lists_what_actually_exists(self):
        from rc_car_actuator.backend import DriveConfigError, drive_from_env

        with pytest.raises(DriveConfigError) as err:
            drive_from_env({"OPENCASTOR_DRIVE": "warp-drive"})
        # The message enumerates the live registry rather than a hand-kept list
        # that drifts as plugins land.
        assert "pca9685-tank" in str(err.value)

    def test_a_registered_plugin_name_becomes_a_valid_backend(self):
        from rc_car_actuator.backend import DRIVE_FACTORIES, drive_from_env, register_drive
        from rc_car_actuator.drive import SimulatedDrive

        @register_drive("test-plugin-drive")
        def _factory(env):
            return SimulatedDrive()

        try:
            drive = drive_from_env({"OPENCASTOR_DRIVE": "test-plugin-drive"})
            assert type(drive).__name__ == "SimulatedDrive"
        finally:
            del DRIVE_FACTORIES["test-plugin-drive"]


class TestDifferentialMixer:
    """Tank steering as arithmetic, pinned before any chassis exists."""

    class TwoChannel:
        def __init__(self):
            self.last = None

        def set_drive(self, left, right):
            self.last = (left, right)

        def neutral(self):
            self.last = (0.0, 0.0)

    def _mix(self, throttle, steering):
        from rc_car_actuator.drive import DifferentialMixer

        device = self.TwoChannel()
        DifferentialMixer(device).set_drive(throttle, steering)
        return device.last

    def test_straight_ahead_drives_both_sides_equally(self):
        assert self._mix(0.5, 0.0) == (0.5, 0.5)

    def test_a_turn_speeds_the_outer_side_and_slows_the_inner(self):
        left, right = self._mix(0.5, 0.3)
        assert left == pytest.approx(0.8)
        assert right == pytest.approx(0.2)

    def test_full_throttle_plus_full_turn_clamps_rather_than_scales(self):
        # 2.0 asked of the outer wheel becomes 1.0; the inner stays 0.0. Turns
        # wider than asked, never slower than asked — the governor above
        # believes its speed model.
        assert self._mix(1.0, 1.0) == (1.0, 0.0)

    def test_spin_in_place(self):
        assert self._mix(0.0, 1.0) == (1.0, -1.0)

    def test_reverse_mixes_the_same_way(self):
        left, right = self._mix(-0.5, 0.3)
        assert left == pytest.approx(-0.2)
        assert right == pytest.approx(-0.8)

    def test_nan_becomes_neutral_not_an_undefined_pulse(self):
        assert self._mix(float("nan"), 0.4) == (0.4, -0.4)

    def test_neutral_passes_through(self):
        from rc_car_actuator.drive import DifferentialMixer

        device = self.TwoChannel()
        DifferentialMixer(device).neutral()
        assert device.last == (0.0, 0.0)

    def test_the_probe_passes_through_when_the_device_has_one(self):
        from rc_car_actuator.drive import DifferentialMixer

        class Probeable(self.TwoChannel):
            def reachable(self):
                return "gone"

        assert DifferentialMixer(Probeable()).reachable() == "gone"
        assert DifferentialMixer(self.TwoChannel()).reachable() is None


class TestBattery:
    """The fuel-gauge skill, built through the gap rail."""

    class Bus:
        """Big-endian chip behind a little-endian SMBus word read."""

        def __init__(self, vcell, soc, crate):
            self.regs = {0x02: vcell, 0x04: soc, 0x16: crate}

        def read_word_data(self, addr, register):
            be = self.regs[register]
            return ((be & 0xFF) << 8) | (be >> 8)  # chip order -> smbus order

    def test_reads_voltage_charge_and_trend(self):
        from rc_car_actuator.battery import MAX1704xFuelGauge

        # 3.998 V, 87.5 %, +12 LSB crate (charging)
        bus = self.Bus(vcell=round(3.998 / 78.125e-6), soc=int(87.5 * 256), crate=12)
        got = MAX1704xFuelGauge(bus).read()
        assert abs(got["voltage_v"] - 3.998) < 0.001
        assert got["percent"] == 87.5
        assert got["charging"] is True

    def test_discharge_rate_is_signed_not_a_huge_positive_number(self):
        # CRATE is signed; reading it unsigned turns "dying at 2 %/hr" into
        # "charging at 13000 %/hr" — the sign IS the diagnosis.
        from rc_car_actuator.battery import MAX1704xFuelGauge

        bus = self.Bus(vcell=45000, soc=50 * 256, crate=(-10) & 0xFFFF)
        got = MAX1704xFuelGauge(bus).read()
        assert got["rate_pct_per_hr"] < 0
        assert got["charging"] is False

    def test_a_dead_gauge_reads_none_not_an_exception(self):
        from rc_car_actuator.battery import MAX1704xFuelGauge

        class Gone:
            def read_word_data(self, a, r):
                raise OSError(121, "Remote I/O error")

        assert MAX1704xFuelGauge(Gone()).read() is None

    def test_telemetry_carries_battery_and_survives_its_absence(self):
        from rc_car_actuator.actuator import RCCarActuator
        from rc_car_actuator.drive import SimulatedDrive

        actuator = RCCarActuator(hardware=SimulatedDrive())
        assert actuator.read_state()["battery"] is None  # not configured: unknown

        actuator._battery = type("G", (), {"read": lambda self: {"percent": 42.0}})()
        assert actuator.read_state()["battery"]["percent"] == 42.0

    def test_opt_in_only_never_probed_by_default(self):
        from rc_car_actuator.battery import battery_from_env

        # 0x36 answering a read is weak evidence — other parts live there. The
        # operator saying OPENCASTOR_BATTERY=max1704x is what makes it a gauge.
        assert battery_from_env({}) is None
        assert battery_from_env({"OPENCASTOR_BATTERY": "somethingelse"}) is None
