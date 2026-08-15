"""MAX1704x fuel gauge — the first skill built through the gap rail.

WHERE THIS CAME FROM. `castor gaps` on the bench rover reported exactly one
gap: "MAX1704x battery fuel gauge on /dev/i2c-1; no declared capability starts
with sensor.*". The operator said build it. That ordering is the rail working
as designed (docs/SKILL-GAPS.md in opencastor-runtime): the robot noticed, a
human allowed, and only then did code get written.

WHY THIS SENSOR EARNS ITS PLACE. The same morning this gap was reported, the
rover's drive pack died mid-bench and announced it in the least useful way
possible: the PCA9685 — powered from the ESC's BEC — vanished off the I2C bus,
and the first symptom anyone saw was a Remote I/O error. A fuel gauge that
rides in telemetry turns that cliff into a slope: the app can show "battery
23% and falling" long before the drive chip browns out.

WHAT IT DELIBERATELY IS NOT. Not a second actuator (a gateway flipped to
multi-actuator 422s every client that doesn't send actuator_name — a bench
scar, not a guess), and not a new tool name. The reading rides inside
`status.report` telemetry, which every client already consumes.

Chip notes (MAX17048/49, the parts on Geekworm-style UPS HATs):
  VCELL 0x02: cell voltage, 78.125 uV/LSB, 16-bit big-endian.
  SOC   0x04: state of charge, 1/256 %/LSB.
  CRATE 0x16: charge/discharge rate, 0.208 %/hr per LSB, SIGNED — the sign is
              the difference between "charging" and "dying", so it must not be
              read as unsigned.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger("rc_car.battery")

_VCELL, _SOC, _CRATE = 0x02, 0x04, 0x16
DEFAULT_ADDRESS = 0x36


class MAX1704xFuelGauge:
    """One battery, read on demand. Never caches: the interesting readings are
    the ones that changed."""

    def __init__(self, bus, address: int = DEFAULT_ADDRESS):  # noqa: ANN001
        self._bus = bus
        self._address = address

    def _word(self, register: int) -> int:
        # The chip is big-endian; SMBus word reads are little-endian. Swap.
        raw = self._bus.read_word_data(self._address, register)
        return ((raw & 0xFF) << 8) | (raw >> 8)

    def read(self) -> dict | None:
        """Voltage, charge and trend — or None when the gauge does not answer.

        None rather than raising, for the same reason `hardware_reachable`
        exists: on this vehicle the WHOLE UPS can be absent (different chassis,
        gauge on the dead rail), and telemetry must degrade to "unknown",
        never take `status.report` down with it. A battery reading is context;
        the drive report it rides in is the product.
        """
        try:
            volts = self._word(_VCELL) * 78.125e-6
            soc = self._word(_SOC) / 256.0
            crate_raw = self._word(_CRATE)
            # Sign-extend by hand: "charging" vs "dying" lives in this bit.
            if crate_raw >= 0x8000:
                crate_raw -= 0x10000
            rate = crate_raw * 0.208
        except Exception as exc:  # noqa: BLE001 - any bus error means absent
            logger.debug("fuel gauge at 0x%02x not answering: %s", self._address, exc)
            return None
        return {
            "voltage_v": round(volts, 3),
            "percent": round(min(100.0, max(0.0, soc)), 1),
            # %/hour; positive while charging. The app renders the verdict —
            # this layer reports the measurement.
            "rate_pct_per_hr": round(rate, 1),
            "charging": rate > 0.5,
        }


def battery_from_env(environ: dict[str, str] | None = None) -> MAX1704xFuelGauge | None:
    """Construct the gauge the environment names, else None.

    Opt-in via OPENCASTOR_BATTERY=max1704x rather than probed-by-default:
    0x36 answering a read is weak evidence (other parts live there), and a
    telemetry field that MIGHT be some other chip's register soup is worse
    than no field. `castor gaps` is where "there seems to be a gauge" belongs;
    this is where "the operator said it IS one" takes effect.
    """
    env = os.environ if environ is None else environ
    if (env.get("OPENCASTOR_BATTERY") or "").strip().lower() != "max1704x":
        return None
    try:
        from smbus2 import SMBus
    except ImportError:
        logger.warning("OPENCASTOR_BATTERY=max1704x but smbus2 is not installed")
        return None
    bus_no = int(env.get("OPENCASTOR_BATTERY_I2C_BUS", "1"))
    address = int(env.get("OPENCASTOR_BATTERY_I2C_ADDRESS", str(DEFAULT_ADDRESS)), 0)
    try:
        return MAX1704xFuelGauge(SMBus(bus_no), address)
    except Exception as exc:  # noqa: BLE001
        logger.warning("fuel gauge bus %d unavailable: %s", bus_no, exc)
        return None
