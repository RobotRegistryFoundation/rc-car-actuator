"""The Maestro backend, tested against a fake serial port.

The protocol has three ways to be silently wrong — units, the high bit, and the
wrong one of the device's two USB ports — and none of them produce an error.
They produce a car that does not move, or moves the wrong amount. Most of the
assertions below are about exactly those.
"""
from __future__ import annotations

import pytest

from rc_car_actuator.drive import Channel
from rc_car_actuator.maestro import (
    MaestroChannels,
    MaestroDrive,
    TARGET_OFF,
)

_SET_TARGET = 0x84
_GET_ERRORS = 0xA1
_GO_HOME = 0xA2


class FakePort:
    """Records every byte written, and can be told what to answer."""

    def __init__(self, to_read: bytes = b""):
        self.written = bytearray()
        self._to_read = to_read
        self.fail_next_write = False

    def write(self, data: bytes) -> int:
        if self.fail_next_write:
            self.fail_next_write = False
            raise OSError("serial write failed")
        self.written.extend(data)
        return len(data)

    def read(self, size: int) -> bytes:
        out, self._to_read = self._to_read[:size], self._to_read[size:]
        return out

    # -- helpers the tests read in terms of ---------------------------------

    def targets(self) -> list[tuple[int, int]]:
        """Every (channel, target) pair sent, in order."""
        out = []
        i = 0
        while i < len(self.written):
            if self.written[i] == _SET_TARGET and i + 3 < len(self.written):
                channel = self.written[i + 1]
                target = self.written[i + 2] | (self.written[i + 3] << 7)
                out.append((channel, target))
                i += 4
            else:
                i += 1
        return out

    def last_us(self, channel: int) -> float | None:
        for ch, target in reversed(self.targets()):
            if ch == channel:
                return target / 4.0
        return None


@pytest.fixture
def port():
    return FakePort()


@pytest.fixture
def drive(port):
    return MaestroDrive(port)


# -- units -------------------------------------------------------------------


def test_targets_are_in_quarter_microseconds():
    # THE UNIT MISTAKE. Sending 1500 instead of 6000 lands at a quarter of the
    # intended pulse — below every servo's minimum, which reads as full lock.
    assert MaestroDrive.target(1500) == 6000
    assert MaestroDrive.target(1000) == 4000
    assert MaestroDrive.target(2000) == 8000


def test_targets_are_clamped_into_what_two_data_bytes_can_carry():
    assert MaestroDrive.target(100_000) == 0x3FFF
    assert MaestroDrive.target(-5) == 0
    assert MaestroDrive.target(float("nan")) == TARGET_OFF


def test_every_data_byte_has_the_high_bit_clear(port, drive):
    # THE PROTOCOL MISTAKE, and the nastiest one: bit 7 marks a COMMAND byte, so
    # a target that leaks into it does not become a wrong position, it becomes a
    # different instruction.
    port.written.clear()
    for throttle in (-1.0, -0.5, 0.0, 0.37, 1.0):
        drive.set_drive(throttle, -throttle)
    i = 0
    while i < len(port.written):
        assert port.written[i] & 0x80, "expected a command byte here"
        i += 1
        while i < len(port.written) and not (port.written[i] & 0x80):
            i += 1


# -- arming ------------------------------------------------------------------


def test_construction_centres_both_channels(port):
    MaestroDrive(port)
    assert port.last_us(0) == pytest.approx(1500, abs=1)
    assert port.last_us(1) == pytest.approx(1500, abs=1)


def test_a_port_that_fails_during_arming_produces_no_drive_object(port):
    port.fail_next_write = True
    with pytest.raises(OSError):
        MaestroDrive(port)


def test_sharing_a_channel_is_refused(port):
    with pytest.raises(ValueError):
        MaestroDrive(port, channels=MaestroChannels(throttle=Channel(2),
                                                    steering=Channel(2)))


# -- commands ----------------------------------------------------------------


def test_full_forward_and_reverse_are_the_hobby_standard_pulses(port, drive):
    drive.set_drive(1.0, 0.0)
    assert port.last_us(0) == pytest.approx(2000, abs=1)
    drive.set_drive(-1.0, 0.0)
    assert port.last_us(0) == pytest.approx(1000, abs=1)


def test_trim_and_inversion_apply_the_same_way_they_do_on_the_pca(port):
    drive = MaestroDrive(port, channels=MaestroChannels(
        throttle=Channel(0, neutral_us=1480, invert=True),
        steering=Channel(1, neutral_us=1520, span_us=300)))
    drive.set_drive(1.0, 1.0)
    assert port.last_us(0) == pytest.approx(980, abs=1)
    assert port.last_us(1) == pytest.approx(1820, abs=1)


def test_nan_becomes_neutral_not_an_arbitrary_pulse(port, drive):
    drive.set_drive(float("nan"), float("nan"))
    assert port.last_us(0) == pytest.approx(1500, abs=1)
    assert port.last_us(1) == pytest.approx(1500, abs=1)


def test_neutral_writes_throttle_before_steering(port, drive):
    port.written.clear()
    drive.neutral()
    channels = [ch for ch, _ in port.targets()]
    assert channels.index(0) < channels.index(1)


def test_neutral_still_straightens_the_wheels_when_stopping_raises(port, drive):
    drive.set_drive(0.5, 1.0)
    port.fail_next_write = True
    with pytest.raises(OSError):
        drive.neutral()
    assert port.last_us(1) == pytest.approx(1500, abs=1)


def test_off_sends_the_stop_pulsing_target_on_both_channels(port, drive):
    port.written.clear()
    drive.off()
    assert sorted(port.targets()) == [(0, TARGET_OFF), (1, TARGET_OFF)]


# -- multiple controllers on one line ----------------------------------------


def test_a_device_number_switches_to_the_pololu_protocol(port):
    drive = MaestroDrive(port, channels=MaestroChannels(
        throttle=Channel(0), steering=Channel(1), device_number=12))
    port.written.clear()
    drive.set_drive(0.0, 0.0)
    assert port.written[0] == 0xAA
    assert port.written[1] == 12
    assert port.written[2] == _SET_TARGET & 0x7F


# -- diagnostics -------------------------------------------------------------


def test_errors_decodes_the_two_seven_bit_bytes():
    # Bit 5 is the serial timeout: 0x20 in the low byte.
    port = FakePort(to_read=bytes([0x20, 0x00]))
    drive = MaestroDrive(port)
    assert drive.errors() == 0x20


def test_a_silent_port_is_reported_as_the_wrong_port_rather_than_hanging():
    # The Maestro presents TWO USB serial devices and only the lower-numbered
    # one takes commands. Opening the other looks exactly like dead hardware, so
    # the error names the actual likely cause.
    port = FakePort(to_read=b"")
    drive = MaestroDrive(port)
    with pytest.raises(OSError) as exc:
        drive.errors()
    assert "command port" in str(exc.value)


def test_go_home_is_a_single_byte(port, drive):
    port.written.clear()
    drive.go_home()
    assert bytes(port.written) == bytes([_GO_HOME])


# -- the contract it has to satisfy ------------------------------------------


def test_it_satisfies_the_same_protocol_the_deadman_drives(port, drive):
    from rc_car_actuator.drive import DriveHardware

    required = {name for name in vars(DriveHardware) if not name.startswith("_")}
    assert required <= set(dir(drive))
