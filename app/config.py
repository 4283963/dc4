from __future__ import annotations

from dataclasses import dataclass, replace

N_HOURS: int = 24


@dataclass(frozen=True)
class BatteryConfig:
    capacity_kwh: float = 100.0
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    max_charge_power_kw: float = 50.0
    max_discharge_power_kw: float = 50.0
    soc_min: float = 0.10
    soc_max: float = 1.00
    dt_hours: float = 1.0
    unmet_load_penalty: float = 1.0e6
    degradation_cost_coeff: float = 5.0e-4


DEFAULT_BATTERY = BatteryConfig()


def build_battery_config(
    *,
    capacity_kwh: float | None = None,
    charge_efficiency: float | None = None,
    discharge_efficiency: float | None = None,
    max_charge_power_kw: float | None = None,
    max_discharge_power_kw: float | None = None,
    soc_min_percent: float | None = None,
    soc_max_percent: float | None = None,
    degradation_cost_coeff: float | None = None,
    base: BatteryConfig = DEFAULT_BATTERY,
) -> BatteryConfig:
    overrides: dict[str, float] = {}
    if capacity_kwh is not None:
        overrides["capacity_kwh"] = capacity_kwh
    if charge_efficiency is not None:
        overrides["charge_efficiency"] = charge_efficiency
    if discharge_efficiency is not None:
        overrides["discharge_efficiency"] = discharge_efficiency
    if max_charge_power_kw is not None:
        overrides["max_charge_power_kw"] = max_charge_power_kw
    if max_discharge_power_kw is not None:
        overrides["max_discharge_power_kw"] = max_discharge_power_kw
    if soc_min_percent is not None:
        overrides["soc_min"] = soc_min_percent / 100.0
    if soc_max_percent is not None:
        overrides["soc_max"] = soc_max_percent / 100.0
    if degradation_cost_coeff is not None:
        overrides["degradation_cost_coeff"] = degradation_cost_coeff
    return replace(base, **overrides)
