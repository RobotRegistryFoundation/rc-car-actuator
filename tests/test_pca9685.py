"""The PCA9685 drive backend, tested against a fake bus.

Every rule here is about a chip that holds its output forever. The tests are
written as the questions somebody would ask standing next to a car on a stand:
did it centre before it could move, does a stop actually stop, and does a
command nobody can interpret become a safe pulse rather than an undefined one.
"""
from __future__ import annotations

import math

import pytest

from rc_car_actuator.pca9685 import (
    DEFAULT_FRAME_HZ,
    NOMINAL_OSCILLATOR_HZ,
    Channel,
    DriveChannels,
    PCA9685Drive,
)

_LED0_ON_L = 0x06
_PRESCALE = 0xFE
_MODE1 = 0x00


class FakeBus:
    """Records every register write, and can be told to fail on cue."""

    def __init__(self):
        self.writes: list[tuple[int, int, int]] = []
        self.registers: dict[int, int] = {}
        self.fail_on_register: int | None = None

    def write_byte_data(self, addr: int, register: int, value: int) -> None:
        if self.fail_on_register is not None and register == self.fail_on_register:
            raise OSError("I2C write failed")
        self.writes.append((addr, register, value))
        self.registers[register] = value

    def read_byte_data(self, addr: int, register: int) -> int:
        return self.registers.get(register, 0)

    # -- helpers the tests read in terms of ---------------------------------

    def pulse_counts(self, channel_index: int) -> int:
        """The current OFF count for a channel, as the chip now holds it."""
        base = _LED0_ON_L + 4 * channel_index
        low = self.registers.get(base + 2, 0)
        high = self.registers.get(base + 3, 0)
        return (high << 8) | low

    def pulse_us(self, channel_index: int, frame_hz: int = DEFAULT_FRAME_HZ) -> float:
        return self.pulse_counts(channel_index) * (1_000_000.0 / frame_hz) / 4096.0


@pytest.fixture
def bus():
    return FakeBus()


#: Arming is a real half-second of held neutral on real hardware. Tests opt out
#: rather than the default being lowered: the delay exists so an ESC recognises
#: the signal, and a suite that quietly made it zero would be testing a driver
#: nobody ships. It also perturbs the deadman's timing in the envelope tests,
#: which is how this was noticed.
NO_ARMING = DriveChannels(throttle=Channel(0), steering=Channel(1), arm_delay_s=0)


@pytest.fixture
def drive(bus):
    return PCA9685Drive(bus, channels=NO_ARMING)


# -- arming ------------------------------------------------------------------


def test_construction_centres_both_channels_before_anything_can_command_motion(bus):
    drive = PCA9685Drive(bus, channels=NO_ARMING)
    assert drive.counts(1500.0) == bus.pulse_counts(0)
    assert bus.pulse_us(0) == pytest.approx(1500, abs=3)
    assert bus.pulse_us(1) == pytest.approx(1500, abs=3)


def test_a_bus_that_fails_during_setup_produces_no_drive_object(bus):
    # If the object existed, something could call set_drive on a chip that was
    # never configured — which out of reset is running at 200 Hz, a rate a servo
    # reads as a permanent full-travel command.
    bus.fail_on_register = _PRESCALE
    with pytest.raises(OSError):
        PCA9685Drive(bus, channels=NO_ARMING)


def test_the_frame_rate_is_written_while_the_chip_is_asleep(bus):
    # The prescale register is write-protected unless SLEEP is set. Written to an
    # awake chip it is silently ignored, and the outputs keep running at the
    # reset default with no error anywhere.
    PCA9685Drive(bus, channels=NO_ARMING)
    order = [reg for _, reg, _ in bus.writes]
    mode1_before_prescale = [
        value for (_, reg, value), position in zip(bus.writes, range(len(bus.writes)))
        if reg == _MODE1 and position < order.index(_PRESCALE)
    ]
    assert mode1_before_prescale, "MODE1 must be written before PRESCALE"
    assert mode1_before_prescale[-1] & 0x10, "SLEEP must be set when PRESCALE is written"


def test_the_chip_is_woken_after_configuration(bus):
    PCA9685Drive(bus, channels=NO_ARMING)
    assert not bus.registers[_MODE1] & 0x10, "SLEEP must be cleared before driving"


def test_sharing_a_channel_between_throttle_and_steering_is_refused(bus):
    # One channel driving both means every steering input is a throttle input,
    # and the first left turn is a launch.
    with pytest.raises(ValueError):
        PCA9685Drive(bus, channels=DriveChannels(arm_delay_s=0, throttle=Channel(0),
                                                 steering=Channel(0)))


# -- the prescale arithmetic -------------------------------------------------


def test_prescale_matches_the_datasheet_for_fifty_hertz():
    # round(25e6 / (4096 * 50)) - 1 == 121, the number every PCA9685 servo
    # example in the world uses.
    assert PCA9685Drive.prescale(50) == 121


def test_prescale_is_clamped_into_the_legal_range():
    # An out-of-range value is not rejected by the chip, it just produces a frame
    # rate nobody asked for — on hardware about to hold it forever.
    assert PCA9685Drive.prescale(10_000) >= 3
    assert PCA9685Drive.prescale(1) <= 255
    with pytest.raises(ValueError):
        PCA9685Drive.prescale(0)


def test_a_measured_oscillator_changes_the_prescale():
    # THE CALIBRATION THAT WILL BITE FIRST. A part running 4% fast emits a 1440
    # us pulse when asked for 1500 — which many ESCs read as a slow crawl in
    # reverse, i.e. a car that moves while commanded to stop.
    fast = PCA9685Drive.prescale(50, oscillator_hz=26_000_000)
    assert fast > PCA9685Drive.prescale(50, oscillator_hz=NOMINAL_OSCILLATOR_HZ)


# -- commands ----------------------------------------------------------------


def test_full_forward_and_full_reverse_are_the_hobby_standard_pulses(bus, drive):
    drive.set_drive(1.0, 0.0)
    assert bus.pulse_us(0) == pytest.approx(2000, abs=3)
    drive.set_drive(-1.0, 0.0)
    assert bus.pulse_us(0) == pytest.approx(1000, abs=3)


def test_steering_and_throttle_are_independent(bus, drive):
    drive.set_drive(0.5, -1.0)
    assert bus.pulse_us(0) == pytest.approx(1750, abs=3)
    assert bus.pulse_us(1) == pytest.approx(1000, abs=3)


def test_a_command_beyond_full_scale_is_clamped_not_wrapped(bus, drive):
    drive.set_drive(5.0, -5.0)
    assert bus.pulse_us(0) == pytest.approx(2000, abs=3)
    assert bus.pulse_us(1) == pytest.approx(1000, abs=3)


def test_nan_becomes_neutral_rather_than_an_undefined_pulse(bus, drive):
    drive.set_drive(float("nan"), float("nan"))
    assert bus.pulse_us(0) == pytest.approx(1500, abs=3)
    assert bus.pulse_us(1) == pytest.approx(1500, abs=3)


def test_the_off_count_never_exceeds_twelve_bits(drive):
    # A pulse longer than the frame does not become a long pulse; the counter
    # wraps and it becomes a short one, turning full forward into something near
    # neutral. Clamping is what keeps that from being expressible.
    assert drive.counts(1_000_000) == 4095
    assert drive.counts(-5) == 0
    assert drive.counts(float("nan")) == 0
    assert all(0 <= drive.counts(us) <= 4095 for us in (0, 1000, 1500, 2000, 25_000))


def test_the_high_byte_only_ever_carries_four_bits(bus, drive):
    # The top nibble of the OFF_H register is the full-off flag. Spilling bit 4
    # of a count into it would silently disable the channel.
    drive.set_drive(1.0, 1.0)
    for base in (_LED0_ON_L, _LED0_ON_L + 4):
        assert bus.registers[base + 3] <= 0x0F


# -- per-vehicle trim --------------------------------------------------------


def test_a_trimmed_neutral_is_what_gets_written(bus):
    # Two identical-looking servo headers disagree about centre: the ESC's is
    # whatever it was taught, the steering servo's is wherever the linkage puts
    # the wheels straight. Hardcoding 1500 for both is how a car creeps at rest
    # and tracks five degrees left.
    drive = PCA9685Drive(bus, channels=DriveChannels(
        arm_delay_s=0,
        throttle=Channel(0, neutral_us=1480),
        steering=Channel(1, neutral_us=1520)))
    drive.neutral()
    assert bus.pulse_us(0) == pytest.approx(1480, abs=3)
    assert bus.pulse_us(1) == pytest.approx(1520, abs=3)


def test_a_narrowed_span_protects_a_linkage_that_binds(bus):
    # A servo pushing against a mechanical bind stalls, heats, and dies.
    drive = PCA9685Drive(bus, channels=DriveChannels(
        arm_delay_s=0,
        throttle=Channel(0),
        steering=Channel(1, span_us=300)))
    drive.set_drive(0.0, 1.0)
    assert bus.pulse_us(1) == pytest.approx(1800, abs=3)


def test_inverting_a_channel_flips_it(bus):
    drive = PCA9685Drive(bus, channels=DriveChannels(
        arm_delay_s=0,
        throttle=Channel(0, invert=True), steering=Channel(1)))
    drive.set_drive(1.0, 0.0)
    assert bus.pulse_us(0) == pytest.approx(1000, abs=3)


# -- stopping ----------------------------------------------------------------


def test_neutral_writes_throttle_before_steering(bus, drive):
    # If the second write fails the car should already be stopped.
    bus.writes.clear()
    drive.neutral()
    registers = [reg for _, reg, _ in bus.writes]
    assert registers.index(_LED0_ON_L + 2) < registers.index(_LED0_ON_L + 6)


def test_neutral_still_straightens_the_wheels_when_stopping_raises(bus, drive):
    # Otherwise a failed stop leaves the wheels locked over for whatever happens
    # next — and what happens next is the deadman writing neutral again.
    drive.set_drive(0.5, 1.0)
    bus.fail_on_register = _LED0_ON_L + 2      # throttle OFF_L
    with pytest.raises(OSError):
        drive.neutral()
    assert bus.pulse_us(1) == pytest.approx(1500, abs=3)


def test_off_sets_the_full_off_bit_for_every_channel(bus, drive):
    drive.off()
    assert bus.registers[0xFD] == 0x10


def test_set_drive_after_neutral_moves_again(bus, drive):
    # Neutral must not latch anything: the deadman stops the car between
    # commands constantly, and a stop that could not be undone would make the
    # first lease the only one.
    drive.neutral()
    drive.set_drive(0.3, 0.0)
    assert bus.pulse_us(0) == pytest.approx(1650, abs=3)


# -- the contract it has to satisfy ------------------------------------------


def test_it_satisfies_the_same_protocol_the_deadman_drives(bus, drive):
    # Checked structurally rather than with isinstance: `DriveHardware` is a
    # plain typing Protocol, and making it runtime-checkable to satisfy a test
    # would be the test changing the shipping contract to suit itself.
    from rc_car_actuator.drive import DriveHardware, SimulatedDrive

    required = {name for name in vars(DriveHardware) if not name.startswith("_")}
    assert required <= set(dir(drive))
    assert required <= set(dir(SimulatedDrive))
    assert math.isclose(bus.pulse_us(0), 1500, abs_tol=3)


# -- pulse-width bounds, ported from castor's RC driver ----------------------


def test_a_mistyped_trim_cannot_emit_a_pulse_that_damages_a_servo(bus):
    # The trims are per-vehicle numbers edited by hand in a config file. A fat
    # fingered 15000 must become a survivable pulse, not a 15 ms one.
    from rc_car_actuator.pca9685 import PULSE_MAX_US, PULSE_MIN_US

    drive = PCA9685Drive(bus, channels=DriveChannels(
        arm_delay_s=0,
        throttle=Channel(0, neutral_us=15000),
        steering=Channel(1, neutral_us=10)))
    drive.neutral()
    assert bus.pulse_us(0) == pytest.approx(PULSE_MAX_US, abs=3)
    assert bus.pulse_us(1) == pytest.approx(PULSE_MIN_US, abs=3)


def test_the_bounds_are_the_same_numbers_castor_uses():
    # Ported, not reinvented. If these drift apart, two drivers on one robot
    # disagree about what is safe to send the same ESC.
    from rc_car_actuator.pca9685 import PULSE_MAX_US, PULSE_MIN_US

    assert (PULSE_MIN_US, PULSE_MAX_US) == (500, 2500)


# -- ESC arming --------------------------------------------------------------


def test_construction_holds_neutral_long_enough_for_an_ESC_to_arm(bus, monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr("rc_car_actuator.pca9685.time.sleep", slept.append)
    PCA9685Drive(bus)                     # default channels => default arm delay
    assert slept and slept[0] == pytest.approx(0.5), \
        "an ESC that never armed ignores everything afterwards"
    assert bus.pulse_us(0) == pytest.approx(1500, abs=3), "and it is NEUTRAL it holds"


def test_reverse_arming_is_OFF_by_default_at_this_layer(bus, monkeypatch):
    # castor's driver defaults it on; this one sits under a 50 ms deadman
    # watchdog and the handshake sleeps. A stop must never wait on an ESC.
    slept: list[float] = []
    monkeypatch.setattr("rc_car_actuator.pca9685.time.sleep", slept.append)
    drive = PCA9685Drive(bus, channels=NO_ARMING)
    drive.set_drive(0.4, 0.0)
    drive.set_drive(-0.4, 0.0)
    assert slept == [], "no handshake unless the operator asked for one"


def test_reverse_arming_sends_neutral_first_when_enabled(bus):
    ch = DriveChannels(arm_delay_s=0, throttle=Channel(0), steering=Channel(1),
                       esc_reverse_arming=True, esc_arm_neutral_ms=1)
    drive = PCA9685Drive(bus, channels=ch)
    drive.set_drive(0.4, 0.0)
    bus.writes.clear()
    drive.set_drive(-0.4, 0.0)
    # Neutral before the reverse pulse: the whole point of the handshake.
    throttles = [t for c, t in _throttle_sequence(bus) if c == 0]
    assert throttles[0] == pytest.approx(1500, abs=3)
    assert throttles[-1] == pytest.approx(1300, abs=3)


def test_only_the_forward_to_reverse_TRANSITION_arms(bus):
    ch = DriveChannels(arm_delay_s=0, throttle=Channel(0), steering=Channel(1),
                       esc_reverse_arming=True, esc_arm_neutral_ms=1)
    drive = PCA9685Drive(bus, channels=ch)
    drive.set_drive(-0.4, 0.0)      # first reverse: arms
    bus.writes.clear()
    drive.set_drive(-0.6, 0.0)      # still reverse: must NOT re-handshake
    throttles = [t for c, t in _throttle_sequence(bus) if c == 0]
    assert throttles == [pytest.approx(1200, abs=3)]


def test_A_STOP_CUTS_THE_HANDSHAKE_SHORT(bus):
    """The rule this whole layer is built around: stopping never waits.

    A reverse handshake can sleep for hundreds of milliseconds. If a deadman
    expiry had to queue behind it, a stop would land late by exactly that much.
    So every pause is a wait on the stop event, and neutral() sets it.
    """
    import threading

    ch = DriveChannels(arm_delay_s=0, throttle=Channel(0), steering=Channel(1),
                       esc_reverse_arming=True, esc_arm_neutral_ms=5000,
                       esc_double_tap_reverse=True)
    drive = PCA9685Drive(bus, channels=ch)
    drive.set_drive(0.4, 0.0)

    done = threading.Event()

    def reverse():
        drive.set_drive(-0.4, 0.0)
        done.set()

    threading.Thread(target=reverse, daemon=True).start()
    import time as _t
    _t.sleep(0.05)                       # let the handshake begin and block
    started = _t.monotonic()
    drive.neutral()                      # the deadman's move
    assert _t.monotonic() - started < 1.0, "neutral() waited on the handshake"
    assert done.wait(timeout=2.0), "the handshake never abandoned itself"
    assert bus.pulse_us(0) == pytest.approx(1500, abs=3), "and it ended at neutral"


def _throttle_sequence(bus):
    """(channel, pulse_us) for every complete channel write, in order."""
    out = []
    pending: dict[int, dict[int, int]] = {}
    for _, reg, val in bus.writes:
        if reg < _LED0_ON_L:
            continue
        channel, offset = divmod(reg - _LED0_ON_L, 4)
        pending.setdefault(channel, {})[offset] = val
        if offset == 3:
            counts = pending[channel].get(2, 0) | (val << 8)
            out.append((channel, counts * (1_000_000.0 / DEFAULT_FRAME_HZ) / 4096.0))
    return out
