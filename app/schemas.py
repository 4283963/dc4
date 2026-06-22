from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config import N_HOURS


class SimulationRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "pv_forecast": [0, 0, 0, 0, 0, 0, 5, 20, 45, 70, 85, 95,
                            90, 80, 65, 40, 15, 2, 0, 0, 0, 0, 0, 0],
            "load_forecast": [22, 20, 19, 18, 20, 25, 30, 35, 40, 42,
                              45, 48, 50, 47, 45, 48, 55, 60, 58, 50, 40, 33, 28, 24],
            "initial_soc": 50.0,
        }
    })

    pv_forecast: List[float] = Field(
        ..., description="每小时预测光伏发电量(kW)，长度必须为 24"
    )
    load_forecast: List[float] = Field(
        ..., description="每小时用户用电负荷(kW)，长度必须为 24"
    )
    initial_soc: float = Field(
        ..., ge=0, le=100, description="电池初始电量 SOC，百分比 0-100"
    )

    battery_capacity_kwh: Optional[float] = Field(default=None, gt=0)
    charge_efficiency: Optional[float] = Field(default=None, gt=0, le=1)
    discharge_efficiency: Optional[float] = Field(default=None, gt=0, le=1)
    max_charge_power_kw: Optional[float] = Field(default=None, gt=0)
    max_discharge_power_kw: Optional[float] = Field(default=None, gt=0)
    soc_min_percent: Optional[float] = Field(default=None, ge=0, le=100)
    soc_max_percent: Optional[float] = Field(default=None, ge=0, le=100)

    @field_validator("pv_forecast", "load_forecast")
    @classmethod
    def _check_length_and_non_negative(cls, v: List[float]) -> List[float]:
        if len(v) != N_HOURS:
            raise ValueError(f"数组长度必须为 {N_HOURS}，当前为 {len(v)}")
        if any(x < 0 for x in v):
            raise ValueError("功率/负荷数值不能为负")
        return v

    @field_validator("soc_min_percent", "soc_max_percent")
    @classmethod
    def _check_soc_range_pair(cls, v):
        return v


class SimulationResponse(BaseModel):
    status: str = Field(..., description="求解状态: optimal / optimal_with_deficit / infeasible")
    message: str = Field(..., description="状态说明")

    soc_curve: List[float] = Field(..., description="每小时末电池 SOC，百分比 0-100，长度 24")
    curtailment: List[float] = Field(..., description="每小时弃光量(kWh)，长度 24")
    total_curtailment: float = Field(..., description="全天总弃光量(kWh)")

    charge_power: List[float] = Field(..., description="每小时电池充电功率(kW)，长度 24")
    discharge_power: List[float] = Field(..., description="每小时电池放电功率(kW)，长度 24")
    unmet_load: List[float] = Field(..., description="每小时未满足负荷(kWh)，长度 24")
    total_unmet_load: float = Field(..., description="全天总未满足负荷(kWh)")

    pv_generation: List[float] = Field(..., description="回显的每小时光伏发电(kW)")
    load: List[float] = Field(..., description="回显的每小时用电负荷(kW)")
    initial_soc: float = Field(..., description="回显的电池初始 SOC(百分比)")
