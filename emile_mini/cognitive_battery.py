# ============================================
# émile-mini "Cognitive Battery"
# - Protocol A: Solo Embodied (sanity)
# - Protocol C1: Context-Switch Adaptation
# - Protocol C2: Memory-Cued Retrieval (uses nav agent)
# Optional: PPO baseline on same wrappers (Gym)
# ============================================

import os, sys, time, json, traceback, importlib, inspect, argparse
from collections import deque
import numpy as np
import pandas as pd

# ------------- Defaults -------------
NEAR_RADIUS = 1
MAX_STEPS_DEFAULT = 200
DT_DEFAULT = 0.01

# ------------- Repo imports (generic) -------------
def _import_or_fail(modname):
    try:
        return importlib.import_module(modname)
    except Exception as e:
        raise ImportError(f"Failed to import {modname}: {e}") from e

_emb_mod = _import_or_fail("emile_mini.embodied_qse_emile")
EmbodiedQSEAgent = getattr(_emb_mod, "EmbodiedQSEAgent", None) or getattr(_emb_mod, "EmbodiedEmileAgent", None)
assert EmbodiedQSEAgent is not None, "EmbodiedQSEAgent not found in emile_mini.embodied_qse_emile"

# Find an environment class in embodied_qse_emile or maze_environment
EmbodiedEnvironment = None
for name, obj in inspect.getmembers(_emb_mod, inspect.isclass):
    if name.lower().endswith(("environment","env")):
        EmbodiedEnvironment = obj
        break
if EmbodiedEnvironment is None:
    try:
        _maze_mod = importlib.import_module("emile_mini.maze_environment")
        for name, obj in inspect.getmembers(_maze_mod, inspect.isclass):
            if name.lower().endswith(("environment","env")):
                EmbodiedEnvironment = obj
                break
    except Exception:
        pass
assert EmbodiedEnvironment is not None, "Could not locate an environment class."

# ------------- Nav imports (memory-cued) -------------
# Uses your nav demo agent for Protocol C2
_nav = _import_or_fail("emile_mini.complete_navigation_system_d")
ClearPathEnvironment = getattr(_nav, "ClearPathEnvironment")
ProactiveEmbodiedQSEAgent = getattr(_nav, "ProactiveEmbodiedQSEAgent")

# ------------- PPO helpers (optional) -------------
def _try_import_sb3():
    try:
        from emile_mini.ppo_nav_baseline import compare_ppo_vs_handcrafted, make_env_kwargs, train_ppo
        return compare_ppo_vs_handcrafted, make_env_kwargs, train_ppo
    except Exception as e:
        return None, None, None

# ------------- Utilities -------------
def _coerce_action(a):
    if isinstance(a, (list, tuple, np.ndarray)):
        arr = np.asarray(a)
        if arr.ndim == 0: return int(arr.item())
        return int(arr.ravel()[0])
    if isinstance(a, np.generic): return int(a.item())
    if isinstance(a, dict):
        if "action" in a: return _coerce_action(a["action"])
        raise TypeError(f"Cannot coerce dict action without 'action' key: {a}")
    return int(a)

def get_env_size(env, default=15):
    for k in ("size","n","width"):
        if hasattr(env,k) and isinstance(getattr(env,k), (int,np.integer)):
            return int(getattr(env,k))
    for k in ("grid","world","map"):
        if hasattr(env,k):
            obj = getattr(env,k)
            try:
                h = len(obj); w = len(obj[0]) if h>0 else h
                return max(h,w)
            except Exception:
                pass
    return default

def get_agent_position(agent, default=(0,0)):
    chains = [
        ("body","state","position"), ("body","position"),
        ("state","position"), ("pos",), ("position",), ("loc",)
    ]
    for chain in chains:
        obj = agent; ok=True
        for part in chain:
            if hasattr(obj, part): obj=getattr(obj, part)
            else: ok=False; break
        if ok and isinstance(obj, (tuple,list,np.ndarray)) and len(obj)>=2:
            x,y = int(obj[0]), int(obj[1]); return (x,y)
    return default

def set_agent_position(agent, xy):
    x,y = int(xy[0]), int(xy[1])
    chains = [
        ("body","state","position"), ("body","position"),
        ("state","position"), ("position",), ("pos",), ("loc",)
    ]
    for chain in chains:
        parent = agent; ok=True
        for part in chain[:-1]:
            if hasattr(parent, part): parent = getattr(parent, part)
            else: ok=False; break
        if ok and hasattr(parent, chain[-1]):
            try:
                setattr(parent, chain[-1], (x,y)); return True
            except Exception: pass
    return False

def get_agent_energy(agent, default=1.0):
    chains = [
        ("body","state","energy"), ("body","energy"),
        ("state","energy"), ("energy",), ("energetics","level"),
    ]
    for chain in chains:
        obj = agent; ok=True
        for part in chain:
            if hasattr(obj, part): obj = getattr(obj, part)
            else: ok=False; break
        if ok and isinstance(obj, (int,float,np.floating)):
            return float(obj)
    return default

def set_agent_energy(agent, val):
    val=float(val)
    chains = [
        ("body","state","energy"), ("body","energy"),
        ("state","energy"), ("energy",), ("energetics","level"),
    ]
    for chain in chains:
        parent = agent; ok=True
        for part in chain[:-1]:
            if hasattr(parent, part): parent = getattr(parent, part)
            else: ok=False; break
        if ok and hasattr(parent, chain[-1]):
            try:
                setattr(parent, chain[-1], val); return True
            except Exception: pass
    return False

def manhattan(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])

def safe_embodied_step(agent, env, dt=DT_DEFAULT):
    for name in ("social_embodied_step","embodied_step","step"):
        fn = getattr(agent, name, None)
        if fn:
            try:
                if "dt" in getattr(fn, "__code__", type("x", (), {"co_varnames":()})()).co_varnames:
                    return fn(env, dt=dt) or {}
                else:
                    return fn(env) or {}
            except Exception:
                traceback.print_exc()
                return {}
    raise RuntimeError("Agent has no step-like method")

def osc_count(history, tail=50):
    if len(history) < 4: return 0
    h = list(history)[-tail:]
    c = 0
    for i in range(3, len(h)):
        if h[i] == h[i-2] and h[i-1] == h[i-3] and h[i] != h[i-1]:
            c += 1
    return c

# ------------- Metrics Accumulation -------------
ROWS = []
def add_row(**kw):
    ROWS.append({k:str(v) for k,v in kw.items()})

def save_tables(prefix):
    df = pd.DataFrame(ROWS)
    csv_path = f"{prefix}_episodes.csv"
    json_path = f"{prefix}_summary.json"
    df.to_csv(csv_path, index=False)

    summary = {}
    by_proto = df.groupby("protocol") if "protocol" in df.columns else []
    for proto, g in by_proto:
        nums = {}
        for col in ("total_reward","steps","adapt_lag","cue_follow_rate",
                    "sigma_ema_mean","sigma_ema_std","qse_surplus_mean",
                    "energy_used","path_len","unique_cells","osc_events",
                    "success_rate","avg_return","avg_steps","avg_spl"):
            if col in g.columns:
                arr = pd.to_numeric(g[col], errors="coerce").dropna().values
                if len(arr)>0:
                    nums[col] = {"mean": float(np.mean(arr)),
                                 "std": float(np.std(arr))}
        summary[proto] = nums

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n💾 Saved:\n- {csv_path}\n- {json_path}")

# ------------- Protocol A -------------
def run_protocol_A(episodes, max_steps, config=None):
    print("🤖 Protocol A: Solo Embodied (Zero Training)")
    rewards=[]
    for ep in range(episodes):
        env = EmbodiedEnvironment()
        agent = EmbodiedQSEAgent(config=config) if config else EmbodiedQSEAgent()
        if hasattr(env, "add_agent"):
            env.add_agent(agent)
        elif hasattr(env, "agents") and isinstance(env.agents, dict):
            env.agents[getattr(agent, "agent_id", f"agent_{ep}")] = agent

        n = get_env_size(env, 15)
        set_agent_position(agent, (2,2))
        set_agent_energy(agent, 1.0)

        pos_hist = deque(maxlen=2000)
        unique = set()
        total_reward=0.0
        for step in range(max_steps):
            out = safe_embodied_step(agent, env)
            r = float(out.get("reward", 0.0)) if isinstance(out, dict) else float(out or 0.0)
            total_reward += r
            p = get_agent_position(agent); pos_hist.append(p); unique.add(p)
            if get_agent_energy(agent) <= 0.05: break

        energy_used = 1.0 - get_agent_energy(agent)
        path_len = 0; last=None
        for p in pos_hist:
            if last is not None: path_len += manhattan(last, p)
            last = p

        # optional signals
        sigma_mean = np.nan; sigma_std = np.nan; qse_surplus_mean = np.nan

        print(f"Episode {ep+1}: reward {total_reward:.2f}, steps {len(pos_hist)}, final energy {get_agent_energy(agent):.2f}")
        rewards.append(total_reward)
        add_row(protocol="A_solo",
                episode=str(ep+1),
                total_reward=total_reward,
                steps=len(pos_hist),
                energy_used=energy_used,
                path_len=path_len,
                unique_cells=len(unique),
                osc_events=osc_count(pos_hist, tail=80),
                sigma_ema_mean=sigma_mean,
                sigma_ema_std=sigma_std,
                qse_surplus_mean=qse_surplus_mean)
    mean, std = float(np.mean(rewards)), float(np.std(rewards))
    print(f"✅ Embodied Émile: {mean:.2f} ± {std:.2f}")
    return mean, std

# ------------- Protocol C1 -------------
def quadrant_targets(n):
    mid = n//2
    return {
        "NW": (1,1),
        "NE": (n-2, 1),
        "SW": (1, n-2),
        "SE": (n-2, n-2),
        "C":  (mid, mid)
    }

def run_protocol_C1(episodes, phases, steps_per_phase, max_steps, config=None):
    print("\n🧭 Protocol C1: Context-Switch Adaptation")
    phase_names = ["NE","SW","C","NW","SE"]
    results=[]
    for ep in range(episodes):
        env = EmbodiedEnvironment()
        agent = EmbodiedQSEAgent(config=config) if config else EmbodiedQSEAgent()
        if hasattr(env, "add_agent"):
            env.add_agent(agent)
        elif hasattr(env, "agents") and isinstance(env.agents, dict):
            env.agents[getattr(agent, "agent_id", f"agentC1_{ep}")] = agent

        n = get_env_size(env, 15)
        targets = quadrant_targets(n)
        plan = phase_names[:phases]
        set_agent_position(agent, (2,2)); set_agent_energy(agent, 1.0)

        pos_hist = deque(maxlen=4000)
        total_reward=0.0

        for ph_idx, ph in enumerate(plan):
            tgt = targets[ph]
            adapted=False; lag=None

            for s in range(steps_per_phase):
                out = safe_embodied_step(agent, env)
                r = float(out.get("reward", 0.0)) if isinstance(out, dict) else float(out or 0.0)
                total_reward += r

                p = get_agent_position(agent); pos_hist.append(p)
                if not adapted and manhattan(p, tgt) <= NEAR_RADIUS:
                    adapted=True; lag=s

                if get_agent_energy(agent) <= 0.05: break

            add_row(protocol="C1_context_switch",
                    episode=str(ep+1),
                    phase=ph,
                    total_reward=total_reward,
                    steps=len(pos_hist),
                    adapt_lag=(lag if lag is not None else np.nan),
                    sigma_delta=np.nan,
                    sigma_ema_mean=np.nan,
                    energy_used=1.0 - get_agent_energy(agent),
                    path_len="",
                    unique_cells="",
                    osc_events=osc_count(pos_hist, tail=80))

        results.append(total_reward)
        print(f"Episode {ep+1}: total reward {total_reward:.2f} across {phases} phases")

    mean, std = float(np.mean(results)), float(np.std(results))
    print(f"✅ C1 total episode reward: {mean:.2f} ± {std:.2f}  (interpret with per-phase adapt_lag)")
    return mean, std

# ------------- Protocol C2 (Memory-Cued Retrieval via nav agent) -------------
def run_protocol_C2_nav(episodes, max_steps, config=None):
    print("\n🧠 Protocol C2: Memory-Cued Retrieval (Nav Agent)")
    results=[]
    quadrants = ["NE","NW","SE","SW","C"]
    for ep in range(episodes):
        env = ClearPathEnvironment(size=20)
        agent = ProactiveEmbodiedQSEAgent(config=config) if config else ProactiveEmbodiedQSEAgent()
        # start at center, face east
        agent.body.state.position = (10, 10)
        agent.body.state.orientation = 0.0

        cue = np.random.choice(quadrants)
        agent.receive_memory_cue({
            "type": "navigation_cue",
            "target_quadrant": cue,
            "instruction": "Navigate",
            "priority": "high"
        })

        # compute target point for distance
        target_pos = agent.memory_goal.target_position
        tgt = tuple(target_pos)

        pos_hist = deque(maxlen=2000)
        unique = set()
        total_reward=0.0
        closer_steps=0
        last_dist = float(np.linalg.norm(np.array(get_agent_position(agent)) - np.array(tgt)))

        for step in range(max_steps):
            out = agent.embodied_step(env)
            # shaped proxy reward from nav: move closer -> reward
            dnow = float(np.linalg.norm(np.array(get_agent_position(agent)) - np.array(tgt)))
            r = (last_dist - dnow)
            total_reward += r

            p = get_agent_position(agent); pos_hist.append(p); unique.add(p)
            if dnow < 2.0:  # success radius consistent with nav module
                break
            if hasattr(agent.body.state, "energy") and get_agent_energy(agent) <= 0.05:
                break

            if dnow < last_dist: closer_steps += 1
            last_dist = dnow

        cue_follow_rate = closer_steps / max(1,len(pos_hist))
        energy_used = (1.0 - get_agent_energy(agent)) if hasattr(agent.body.state, "energy") else np.nan
        path_len = 0; last=None
        for q in pos_hist:
            if last is not None: path_len += manhattan(last, q)
            last = q

        print(f"Episode {ep+1}: cue={cue}, reward {total_reward:.2f}, cue_follow_rate={cue_follow_rate:.2f}")
        results.append(total_reward)

        add_row(protocol="C2_memory_cued",
                episode=str(ep+1),
                cue=cue,
                total_reward=total_reward,
                steps=len(pos_hist),
                cue_follow_rate=cue_follow_rate,
                energy_used=energy_used,
                path_len=path_len,
                unique_cells=len(unique),
                osc_events=osc_count(pos_hist, tail=80),
                sigma_ema_mean=np.nan,
                qse_surplus_mean=np.nan)

    mean, std = float(np.mean(results)), float(np.std(results))
    print(f"✅ C2 (nav) total episode reward: {mean:.2f} ± {std:.2f}  (watch cue_follow_rate)")
    return mean, std

# ------------- Helper to build QSEConfig from multimodal flags -------------
def _build_multimodal_config(multimodal: bool = False, modality_scale: float = None):
    """Helper to build QSEConfig with multimodal settings from CLI flags."""
    from emile_mini.config import QSEConfig
    cfg = QSEConfig()
    if multimodal:
        cfg.MULTIMODAL_ENABLED = True
        if modality_scale is not None:
            cfg.MODALITY_INFLUENCE_SCALE = modality_scale
    return cfg

# ------------- Main -------------
def main(
    episodes_a: int,
    episodes_c1: int,
    phases_c1: int,
    steps_per_phase: int,
    episodes_c2: int,
    max_steps: int,
    seed: int,
    prefix: str,
    run_ppo: bool,
    ppo_timesteps: int,
    size: int,
    max_steps_nav: int,
    obstacles: float,
    start_mode: str,
    quadrant: str,
    multimodal: bool = False,
    modality_scale: float = None,
):
    np.random.seed(seed)

    # Build multimodal config from flags
    cfg = _build_multimodal_config(multimodal, modality_scale)

    a_mean, a_std = run_protocol_A(episodes=episodes_a, max_steps=max_steps, config=cfg)
    c1_mean, c1_std = run_protocol_C1(episodes=episodes_c1, phases=phases_c1, steps_per_phase=steps_per_phase, max_steps=max_steps, config=cfg)
    c2_mean, c2_std = run_protocol_C2_nav(episodes=episodes_c2, max_steps=max_steps, config=cfg)

    # Optional PPO baseline & comparison on NavEnv
    compare_ppo_vs_handcrafted, make_env_kwargs, train_ppo = _try_import_sb3()
    if run_ppo and compare_ppo_vs_handcrafted is not None:
        kw = make_env_kwargs(size=size, max_steps=max_steps_nav, obstacle_density=obstacles,
                             quadrant=quadrant, start_mode=start_mode, seed=seed)
        stats = compare_ppo_vs_handcrafted(n_episodes=200, timesteps=ppo_timesteps,
                                           model_path="ppo_nav.zip", **kw)
        # record summary rows
        add_row(protocol="PPO_compare", episode="summary",
                success_rate=stats["ppo"]["success_rate"],
                avg_return=stats["ppo"]["avg_return"],
                avg_steps=stats["ppo"]["avg_steps"],
                avg_spl=stats["ppo"]["avg_spl"])
        add_row(protocol="Handcrafted_compare", episode="summary",
                success_rate=stats["handcrafted"]["success_rate"],
                avg_return=stats["handcrafted"]["avg_return"],
                avg_steps=stats["handcrafted"]["avg_steps"],
                avg_spl=stats["handcrafted"]["avg_spl"])
        print("\n=== COMPARISON ===")
        for k in ("ppo","handcrafted"):
            s = stats[k]
            print(f"{k.upper():>12}: success={s['success_rate']:.3f}  "
                  f"ret={s['avg_return']:.3f}  steps={s['avg_steps']:.1f}  SPL={s['avg_spl']:.3f}")
    elif run_ppo:
        print("⚠️ PPO comparison requested but stable-baselines3 not found. "
              "Install with: pip install stable-baselines3 gymnasium")

    print("\n📊 SUMMARY (first rows)")
    df = pd.DataFrame(ROWS)
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(df.head(10))

    save_tables(prefix=prefix)
    return 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser("emile-mini cognitive battery")
    ap.add_argument("--episodes-a", type=int, default=5)
    ap.add_argument("--episodes-c1", type=int, default=3)
    ap.add_argument("--phases-c1", type=int, default=3)
    ap.add_argument("--steps-per-phase", type=int, default=150)
    ap.add_argument("--episodes-c2", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prefix", type=str, default="emile_cognitive_battery")

    # PPO flags (optional)
    ap.add_argument("--ppo", action="store_true")
    ap.add_argument("--ppo-timesteps", type=int, default=50000)
    ap.add_argument("--size", type=int, default=20)
    ap.add_argument("--max-steps-nav", type=int, default=80)
    ap.add_argument("--obstacles", type=float, default=0.15)
    ap.add_argument("--start-mode", choices=["random","center"], default="random")
    ap.add_argument("--quadrant", choices=["NE","NW","SE","SW","C"], default="NE")

    args = ap.parse_args()
    sys.exit(main(
        episodes_a=args.episodes_a,
        episodes_c1=args.episodes_c1,
        phases_c1=args.phases_c1,
        steps_per_phase=args.steps_per_phase,
        episodes_c2=args.episodes_c2,
        max_steps=args.max_steps,
        seed=args.seed,
        prefix=args.prefix,
        run_ppo=args.ppo,
        ppo_timesteps=args.ppo_timesteps,
        size=args.size,
        max_steps_nav=args.max_steps_nav,
        obstacles=args.obstacles,
        start_mode=args.start_mode,
        quadrant=args.quadrant,
    ))
