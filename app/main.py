from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI, HTTPException

from .config import build_battery_config
from .schemas import SimulationRequest, SimulationResponse
from .simulator import run_simulation

app = FastAPI(
    title="离网微电网运行仿真引擎",
    description=(
        "光伏 + 储能电池 + 充电桩离网微电网的 24 小时最优充放电策略计算。"
        "基于 NumPy / SciPy 线性规划，在保证用电、禁止电池过充过放的前提下，"
        "最小化弃光量。"
    ),
    version="1.0.0",
)


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok"}


@app.post(
    "/api/v1/simulation/run",
    response_model=SimulationResponse,
    tags=["simulation"],
    summary="运行一天(24h)微电网储能调度仿真",
)
def run_simulation_endpoint(req: SimulationRequest) -> SimulationResponse:
    battery = build_battery_config(
        capacity_kwh=req.battery_capacity_kwh,
        charge_efficiency=req.charge_efficiency,
        discharge_efficiency=req.discharge_efficiency,
        max_charge_power_kw=req.max_charge_power_kw,
        max_discharge_power_kw=req.max_discharge_power_kw,
        soc_min_percent=req.soc_min_percent,
        soc_max_percent=req.soc_max_percent,
    )
    try:
        result = run_simulation(
            req.pv_forecast,
            req.load_forecast,
            req.initial_soc,
            battery,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return SimulationResponse(**asdict(result))
