"""Choosing which drive layer the actuator actually gets.

WHY THIS IS A SEPARATE FILE AND NOT A FEW `if`s IN `__init__`. The gateway
constructs actuators from an entry point with NO ARGUMENTS (`cls()`), so
`RCCarActuator(hardware=...)` — the clean seam the tests use — is unreachable
in production. Without something here, a car wired to a PCA9685 would be driven
by `SimulatedDrive` forever, and the only symptom would be wheels that never
turn while every receipt reports a successful drive.

TWO RULES, PULLING IN OPPOSITE DIRECTIONS, AND BOTH ARE LOAD-BEARING:

  1. The DEFAULT IS SIMULATED, always. Real hardware is only ever reached by
     someone explicitly asking for it. Getting hardware you did not ask for is
     how a bench test becomes a car driving off a table.

  2. ASKING FOR HARDWARE AND SILENTLY GETTING SIMULATION IS ALSO A FAILURE, and
     this file refuses to do it. A fallback would be "safe" in the narrow sense
     that nothing moves, and would cost an afternoon: the wheels sit still, the
     receipts all say ALLOW with a granted lease, the logs look perfect, and the
     wiring gets torn apart looking for a fault that is a missing Python package.
     An explicit request that cannot be honoured raises.

Configuration is read from the environment because that is what a systemd unit
can set (see the rover's `gateway-policy.env`), and because per-vehicle
calibration — the trims below — belongs next to the vehicle rather than in a
package.
"""
from __future__ import annotations

import logging
import os

from .drive import DriveHardware, SimulatedDrive

logger = logging.getLogger("rc_car.backend")

#: Which drive layer to construct. Anything other than a known name is an error
#: rather than a fallback: a typo'd `pca8695` must not quietly become simulation.
ENV_BACKEND = "OPENCASTOR_DRIVE"


class DriveConfigError(RuntimeError):
    """A drive backend was explicitly requested and could not be provided."""


# Every reader takes the environment explicitly rather than reaching for
# os.environ. Threading it through is not ceremony: the first version of this
# file accepted an `environ` argument at the top and then ignored it three
# functions down, so a test that set a malformed value watched the default sail
# past and failed on an unrelated I2C error instead. A config reader that reads
# a different environment from the one it was handed is a config reader whose
# validation cannot be tested.
def _float(env, name: str, default: float) -> float:  # noqa: ANN001
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise DriveConfigError(f"{name}={raw!r} is not a number") from exc


def _int(env, name: str, default: int) -> int:  # noqa: ANN001
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        # base=0 so 0x40 works, which is how every PCA9685 address is written
        # in every datasheet, breakout silkscreen, and i2cdetect output.
        return int(raw, 0)
    except ValueError as exc:
        raise DriveConfigError(f"{name}={raw!r} is not an integer") from exc


def _bool(env, name: str, default: bool = False) -> bool:  # noqa: ANN001
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def drive_from_env(environ: dict[str, str] | None = None) -> DriveHardware:
    """Construct the drive layer named by the environment.

    Returns `SimulatedDrive` when nothing is set, which is the case that must
    stay boring.
    """
    env = os.environ if environ is None else environ
    name = (env.get(ENV_BACKEND) or "simulated").strip().lower()

    if name in {"simulated", "sim", "none"}:
        return SimulatedDrive()

    if name == "pca9685":
        return _pca9685_from_env(env)

    if name == "maestro":
        return _maestro_from_env(env)

    if name == "pigpio":
        return _pigpio_from_env(env)

    raise DriveConfigError(
        f"{ENV_BACKEND}={name!r} is not a known drive backend "
        f"(simulated, pca9685, maestro, pigpio)")


def _channel_from_env(env, prefix: str, index_default: int):  # noqa: ANN001, ANN201
    """The per-vehicle trim for one channel, shared by both PWM backends.

    Identical for a PCA9685 and a Maestro because the measurements are about the
    VEHICLE — where the ESC sits still, where the linkage points the wheels
    straight, how far it can travel before it binds — and not about the chip.
    """
    from .drive import Channel

    return Channel(
        index=_int(env, f"OPENCASTOR_DRIVE_{prefix}_CHANNEL", index_default),
        neutral_us=_float(env, f"OPENCASTOR_DRIVE_{prefix}_NEUTRAL_US", 1500.0),
        span_us=_float(env, f"OPENCASTOR_DRIVE_{prefix}_SPAN_US", 500.0),
        invert=_bool(env, f"OPENCASTOR_DRIVE_{prefix}_INVERT"),
    )


def _maestro_from_env(env) -> DriveHardware:  # noqa: ANN001
    from .maestro import MaestroChannels, MaestroDrive

    device = env.get("OPENCASTOR_DRIVE_MAESTRO_DEVICE")
    channels = MaestroChannels(
        throttle=_channel_from_env(env, "THROTTLE", 0),
        steering=_channel_from_env(env, "STEERING", 1),
        device_number=_int(env, "OPENCASTOR_DRIVE_MAESTRO_DEVICE", 0) if device else None,
        steering_speed=_int(env, "OPENCASTOR_DRIVE_STEERING_SPEED", 0),
    )
    port_name = env.get("OPENCASTOR_DRIVE_SERIAL_PORT") or "/dev/ttyACM0"

    try:
        import serial
    except ImportError as exc:
        raise DriveConfigError(
            "OPENCASTOR_DRIVE=maestro needs pyserial (pip install "
            "'rc-car-actuator[maestro]'). Refusing to fall back to simulation."
        ) from exc

    try:
        port = serial.Serial(port_name, 115200, timeout=0.2)
    except Exception as exc:  # serial.SerialException and friends
        raise DriveConfigError(
            f"cannot open {port_name} ({exc}). The Maestro presents TWO serial "
            f"devices and only the LOWER-numbered one is the command port; "
            f"check `ls /dev/ttyACM*` and that the user is in the dialout group."
        ) from exc

    logger.warning(
        "REAL DRIVE HARDWARE: Pololu Maestro on %s, throttle ch%d, steering ch%d. "
        "The serial-timeout failsafe is a DEVICE setting this code cannot read — "
        "verify it by unplugging on a stand, not by trusting this line. "
        "Keep the wheels off the ground until the hardware stop is fitted.",
        port_name, channels.throttle.index, channels.steering.index)
    return MaestroDrive(port, channels=channels)


def _pca9685_from_env(env) -> DriveHardware:  # noqa: ANN001
    from .pca9685 import DriveChannels, PCA9685Drive

    channels = DriveChannels(
        throttle=_channel_from_env(env, "THROTTLE", 0),
        steering=_channel_from_env(env, "STEERING", 1),
        frame_hz=_int(env, "OPENCASTOR_DRIVE_FRAME_HZ", 50),
        # See `DriveChannels.oscillator_hz`: the chip's 25 MHz is a cheap RC
        # oscillator, and every pulse it emits is scaled by the real figure.
        oscillator_hz=_int(env, "OPENCASTOR_DRIVE_OSCILLATOR_HZ", 25_000_000),
    )
    address = _int(env, "OPENCASTOR_DRIVE_I2C_ADDRESS", 0x40)
    bus_number = _int(env, "OPENCASTOR_DRIVE_I2C_BUS", 1)

    try:
        import smbus2
    except ImportError as exc:
        raise DriveConfigError(
            "OPENCASTOR_DRIVE=pca9685 needs smbus2 (pip install 'rc-car-actuator[pca9685]'). "
            "Refusing to fall back to simulation: silently not moving is the "
            "failure that costs an afternoon of checking wiring."
        ) from exc

    try:
        bus = smbus2.SMBus(bus_number)
    except OSError as exc:
        raise DriveConfigError(
            f"cannot open I2C bus {bus_number} ({exc}). Is I2C enabled "
            f"(raspi-config → Interface Options → I2C) and is the user in the "
            f"i2c group?"
        ) from exc

    logger.warning(
        "REAL DRIVE HARDWARE: PCA9685 at 0x%02x on i2c-%d, throttle ch%d, steering ch%d. "
        "Keep the wheels off the ground until the hardware stop is fitted.",
        address, bus_number, channels.throttle.index, channels.steering.index)
    return PCA9685Drive(bus, address=address, channels=channels)


def _pigpio_from_env(env) -> DriveHardware:  # noqa: ANN001
    from .drive import PWMDrive

    try:
        import pigpio
    except ImportError as exc:
        raise DriveConfigError(
            "OPENCASTOR_DRIVE=pigpio needs the pigpio package "
            "(pip install 'rc-car-actuator[hardware]')") from exc

    pi = pigpio.pi()
    if not pi.connected:
        raise DriveConfigError(
            "pigpio daemon is not reachable — is pigpiod running?")
    return PWMDrive(pi,
                    throttle_gpio=_int(env, "OPENCASTOR_DRIVE_THROTTLE_GPIO", 18),
                    steering_gpio=_int(env, "OPENCASTOR_DRIVE_STEERING_GPIO", 19))
