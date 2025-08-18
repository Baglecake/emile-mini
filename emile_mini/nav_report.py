import os, math, json
from typing import List, Sequence, Dict, Any
import numpy as np
import pandas as pd

from .ppo_nav_baseline import compare_ppo_vs_handcrafted, make_env_kwargs

def _ci95(p: float, n: int):
    if n <= 0 or p is None or np.isnan(p):
        return (float("nan"), float("nan"))
    se = math.sqrt(p * (1 - p) / n)
    lo = max(0.0, p - 1.96 * se)
    hi = min(1.0, p + 1.96 * se)
    return (lo, hi)

def _rowpack(group: str, value, label: str, stats: Dict[str, Any], episodes: int):
    # stats keys: success_rate, avg_return, avg_steps, avg_spl
    lo, hi = _ci95(stats["success_rate"], episodes)
    return {
        "group": group,
        "value": value,
        "model": label,  # "PPO baseline" or "ÉMILE-MINI"
        "episodes": episodes,
        "success": float(stats["success_rate"]),
        "success_CI_low_95": lo,
        "success_CI_high_95": hi,
        "SPL": float(stats.get("avg_spl", float("nan"))),
        "avg_steps": float(stats.get("avg_steps", float("nan"))),
        "avg_return": float(stats.get("avg_return", float("nan"))),
    }

def run_report(
    quadrants: Sequence[str] = ("NE","NW","SE","SW","C"),
    obstacles: float = 0.20,                    # used for the quadrant sweep
    densities: Sequence[float] = (0.10,0.20,0.30),
    density_quadrant: str = "NE",               # used for the density sweep
    episodes: int = 400,                        # per condition
    timesteps: int = 50000,                     # if model missing
    size: int = 20,
    max_steps: int = 80,
    start_mode: str = "random",
    seed: int = 123,
    model: str = "ppo_nav.zip",
    # reward shaping knobs (must match NavEnv)
    progress_k: float = 0.5,
    step_cost: float = 0.02,
    turn_penalty: float = 0.01,
    collision_penalty: float = 0.10,
    success_bonus: float = 2.0,
    prefix: str = "nav_report",
) -> Dict[str, str]:
    rows = []

    # ---- Quadrant sweep (fixed obstacle density) ----
    for q in quadrants:
        kw = make_env_kwargs(
            size=size, max_steps=max_steps, obstacle_density=obstacles,
            quadrant=q, start_mode=start_mode, seed=seed,
            progress_k=progress_k, step_cost=step_cost, turn_penalty=turn_penalty,
            collision_penalty=collision_penalty, success_bonus=success_bonus
        )
        out = compare_ppo_vs_handcrafted(n_episodes=episodes, timesteps=timesteps,
                                         model_path=model, **kw)
        rows.append(_rowpack("quadrant", q, "PPO baseline", out["ppo"], episodes))
        rows.append(_rowpack("quadrant", q, "ÉMILE-MINI", out["handcrafted"], episodes))

    # ---- Density sweep (fixed quadrant) ----
    for d in densities:
        kw = make_env_kwargs(
            size=size, max_steps=max_steps, obstacle_density=float(d),
            quadrant=density_quadrant, start_mode=start_mode, seed=seed,
            progress_k=progress_k, step_cost=step_cost, turn_penalty=turn_penalty,
            collision_penalty=collision_penalty, success_bonus=success_bonus
        )
        out = compare_ppo_vs_handcrafted(n_episodes=episodes, timesteps=timesteps,
                                         model_path=model, **kw)
        rows.append(_rowpack("obstacles", float(d), "PPO baseline", out["ppo"], episodes))
        rows.append(_rowpack("obstacles", float(d), "ÉMILE-MINI", out["handcrafted"], episodes))

    df = pd.DataFrame(rows).sort_values(["group","value","model"])
    csv_path = f"{prefix}_results.csv"
    json_path = f"{prefix}_summary.json"
    df.to_csv(csv_path, index=False)

    # compact summary: means by group & model
    summary: Dict[str, Any] = {}
    for grp, gdf in df.groupby(["group","value","model"]):
        g = {
            "episodes": int(gdf["episodes"].iloc[0]),
            "success_mean": float(gdf["success"].mean()),
            "SPL_mean": float(gdf["SPL"].mean()),
            "steps_mean": float(gdf["avg_steps"].mean()),
            "return_mean": float(gdf["avg_return"].mean()),
        }
        summary.setdefault(grp[0], {}).setdefault(str(grp[1]), {})[grp[2]] = g

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n💾 Saved:\n- {csv_path}\n- {json_path}")
    return {"csv": csv_path, "json": json_path}
