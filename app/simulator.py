from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from scipy.optimize import linprog

from .config import BatteryConfig, DEFAULT_BATTERY, N_HOURS


@dataclass
class SimulationResult:
    status: str
    message: str
    soc_curve: List[float]
    curtailment: List[float]
    total_curtailment: float
    charge_power: List[float]
    discharge_power: List[float]
    unmet_load: List[float]
    total_unmet_load: float
    pv_generation: List[float]
    load: List[float]
    initial_soc: float


def run_simulation(
    pv_forecast: Sequence[float],
    load_forecast: Sequence[float],
    initial_soc_percent: float,
    battery: Optional[BatteryConfig] = None,
) -> SimulationResult:
    """运行 24 小时离网微电网储能调度优化。

    线性规划模型说明
    ----------------
    决策变量（每小时一个，共 24 小时）：
        C[t]  电池充电功率 (kW)，仅在“光伏富余小时”从富余光伏充入
        D[t]  电池放电功率 (kW)，仅在“光伏不足小时”放电补足负荷
        Q[t]  弃光功率 (kW)，光伏富余但电池已满/功率受限而浪费
        U[t]  未满足负荷 (kW)，作为“保供电”安全松弛

    关键设计：光伏与负荷是已知输入，故可按 net = PV - Load 的符号
        预先划分每一小时：
          * 富余小时(PV>=Load)：只允许充电(C)、弃光(Q)，禁止放电(D=U=0)
          * 不足小时(PV< Load)：只允许放电(D)、未满足(U)，禁止充电(C=Q=0)
        这样从结构上杜绝“同时充放”的退化解（避免通过往返效率损失
        把弃光伪装成损耗、从而虚降弃光量）。

    目标函数：min sum(Q[t]) + BIG * sum(U[t]) - eps * sum((N-t) * C[t])
        优先级：保证“用电”尽量满足(BIG) > 最小化“弃光”(1) > 尽早充电(eps)。
        第三项为极小权重，仅在弃光相同的解中挑选“先充满电池、满了才弃光”
        的直观调度，绝不影响弃光最小化与保供电。

    约束：
        1) 功率平衡：C - D + Q - U = PV - Load            (等式)
        2) 荷电状态递推：SOC[t+1] = SOC[t]
           + (eta_c*C - D/eta_d)*dt / Capacity
        3) 安全区间：soc_min <= SOC <= soc_max，禁止过充过放
        4) 设备功率上下限：0<=C<=Pc_max, 0<=D<=Pd_max
    """
    battery = battery if battery is not None else DEFAULT_BATTERY

    pv = np.asarray(pv_forecast, dtype=float).reshape(-1)
    load = np.asarray(load_forecast, dtype=float).reshape(-1)
    if pv.shape[0] != N_HOURS or load.shape[0] != N_HOURS:
        raise ValueError(f"光伏/负荷数组长度必须为 {N_HOURS}")
    n = N_HOURS

    soc0 = float(initial_soc_percent) / 100.0
    Cap = battery.capacity_kwh
    eta_c = battery.charge_efficiency
    eta_d = battery.discharge_efficiency
    inv_eta_d = 1.0 / eta_d
    P_c_max = battery.max_charge_power_kw
    P_d_max = battery.max_discharge_power_kw
    soc_min = battery.soc_min
    soc_max = battery.soc_max
    dt = battery.dt_hours
    BIG = battery.unmet_load_penalty

    if soc0 < soc_min - 1e-9 or soc0 > soc_max + 1e-9:
        raise ValueError(
            f"初始 SOC {initial_soc_percent:.2f}% 超出安全区间 "
            f"[{soc_min * 100:.1f}%, {soc_max * 100:.1f}%]"
        )

    nv = 4 * n
    tie_break = 1.0e-4 * (n - np.arange(n))
    c = np.concatenate([
        -tie_break,
        np.zeros(n),
        np.ones(n),
        np.full(n, BIG),
    ])

    A_eq = np.zeros((n, nv))
    A_eq[:, 0:n] = np.eye(n)
    A_eq[:, n:2 * n] = -np.eye(n)
    A_eq[:, 2 * n:3 * n] = np.eye(n)
    A_eq[:, 3 * n:4 * n] = -np.eye(n)
    b_eq = (pv - load).astype(float)

    idx = np.arange(n)
    mask = (idx[:, None] >= idx[None, :]).astype(float)
    zeros_nn = np.zeros((n, n))

    rhs_upper = Cap * (soc_max - soc0) / dt
    rhs_lower = Cap * (soc0 - soc_min) / dt

    upper_soc = np.hstack([eta_c * mask, -inv_eta_d * mask, zeros_nn, zeros_nn])
    lower_soc = np.hstack([-eta_c * mask, inv_eta_d * mask, zeros_nn, zeros_nn])
    A_ub = np.vstack([upper_soc, lower_soc])
    b_ub = np.concatenate([
        np.full(n, rhs_upper),
        np.full(n, rhs_lower),
    ])

    surplus = pv >= load
    surplus_amt = np.maximum(0.0, pv - load)
    deficit_amt = np.maximum(0.0, load - pv)
    c_ub = np.where(surplus, P_c_max, 0.0)
    d_ub = np.where(surplus, 0.0, P_d_max)
    q_ub = np.where(surplus, surplus_amt, 0.0)
    u_ub = np.where(surplus, 0.0, deficit_amt)
    bounds = (
        [(0.0, float(v)) for v in c_ub]
        + [(0.0, float(v)) for v in d_ub]
        + [(0.0, float(v)) for v in q_ub]
        + [(0.0, float(v)) for v in u_ub]
    )

    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    empty = [0.0] * n
    if res.x is None or not res.success:
        return SimulationResult(
            status="infeasible",
            message=f"优化求解失败，可能存在输入不可行: {res.message}",
            soc_curve=[float(initial_soc_percent)] + empty[1:],
            curtailment=empty,
            total_curtailment=0.0,
            charge_power=empty,
            discharge_power=empty,
            unmet_load=empty,
            total_unmet_load=0.0,
            pv_generation=[round(float(v), 4) for v in pv],
            load=[round(float(v), 4) for v in load],
            initial_soc=float(initial_soc_percent),
        )

    x = res.x
    C = np.clip(x[0:n], 0.0, None)
    D = np.clip(x[n:2 * n], 0.0, None)
    Q = np.clip(x[2 * n:3 * n], 0.0, None)
    U = np.clip(x[3 * n:4 * n], 0.0, None)

    delta_soc = (eta_c * C - inv_eta_d * D) * dt / Cap
    soc_end = np.clip(soc0 + np.cumsum(delta_soc), soc_min, soc_max)

    curtailment_kwh = Q * dt
    unmet_kwh = U * dt
    total_curt = float(np.sum(curtailment_kwh))
    total_unmet = float(np.sum(unmet_kwh))

    tol = 1e-4
    if total_unmet > tol:
        status = "optimal_with_deficit"
        message = (
            f"优化完成，但存在 {total_unmet:.3f} kWh 负荷无法满足"
            f"（储能功率或容量不足）；在该前提下弃光已最小化。"
        )
    else:
        total_unmet = 0.0
        unmet_kwh = np.zeros(n)
        status = "optimal"
        message = "优化完成：负荷全部满足，弃光量已最小化，SOC 始终位于安全区间。"

    return SimulationResult(
        status=status,
        message=message,
        soc_curve=[round(float(v) * 100.0, 4) for v in soc_end],
        curtailment=[round(float(v), 4) for v in curtailment_kwh],
        total_curtailment=round(total_curt, 4),
        charge_power=[round(float(v), 4) for v in C],
        discharge_power=[round(float(v), 4) for v in D],
        unmet_load=[round(float(v), 4) for v in unmet_kwh],
        total_unmet_load=round(total_unmet, 4),
        pv_generation=[round(float(v), 4) for v in pv],
        load=[round(float(v), 4) for v in load],
        initial_soc=float(initial_soc_percent),
    )
