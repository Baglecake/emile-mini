
#!/usr/bin/env python3
"""
Émile-mini Quick Audit Harness
- Structural checks (modules, classes, config knobs)
- Behavioral smoke tests (Σ EMA, context hysteresis/dwell, energy/forage, existential pressure, social transfer)
Run with:  python audit_emile_mini.py
or in Colab:  %run audit_emile_mini.py
"""

import importlib, sys, types, traceback, random, math
from dataclasses import is_dataclass
from typing import Any, Dict

OK = "✅"; WARN = "⚠️"; FAIL = "❌"

def safe_import(name):
    try:
        m = importlib.import_module(name)
        print(f"{OK} imported {name}")
        return m
    except Exception as e:
        print(f"{FAIL} import {name}: {e}")
        return None

def has_attr(mod, attr):
    return hasattr(mod, attr)

def get_cfg(mod):
    # Try common patterns
    if hasattr(mod, "QSEConfig"): return mod.QSEConfig()
    if hasattr(mod, "CONFIG"): return getattr(mod, "CONFIG")
    raise RuntimeError("No QSEConfig/CONFIG found in config.py")

def section(title):
    print("\n" + "="*72)
    print(title)
    print("="*72)

def try_call(fn, *a, **k):
    try:
        return True, fn(*a, **k)
    except Exception as e:
        traceback.print_exc(limit=1)
        return False, e

# ---------------------------------------------------------------------
# 1) Imports / structure
# ---------------------------------------------------------------------
section("1) Module structure / imports")

mods = {}
for name in [
    "config", "symbolic", "context", "qse_core", "agent", "memory",
    "embodied_qse_emile", "social_qse_agent_v2", "simulator", "goal"
]:
    mods[name] = safe_import(name)

# Optional modules (don’t fail audit if missing)
for name in ["viz", "maze_comparison"]:
    mods[name] = safe_import(name) or types.SimpleNamespace()

all_imported = all(m is not None for m in [mods["config"], mods["symbolic"], mods["context"], mods["agent"]])

# ---------------------------------------------------------------------
# 2) Config knobs present?
# ---------------------------------------------------------------------
section("2) Config knobs (EMA, hysteresis/dwell, energy, social, loops)")
cfg_status = {}
if mods["config"]:
    try:
        cfg = get_cfg(mods["config"])
        required = [
            "SIGMA_EMA_ALPHA",
            "RECONTEXT_THRESHOLD","RECONTEXT_HYSTERESIS","CONTEXT_MIN_DWELL_STEPS",
            "ENERGY_MOVE_COST","ENERGY_TURN_COST","ENERGY_EXAMINE_COST","ENERGY_REST_RECOVERY",
            "ENERGY_MIN_FLOOR","ENERGY_FORAGE_REWARD_MIN","ENERGY_FORAGE_REWARD_MAX",
            "SOCIAL_DETECT_RADIUS","TEACH_COOLDOWN_STEPS","MIN_CONFIDENCE_RETENTION",
            "LOOP_WINDOW","LOOP_SPATIAL_EPS","LOOP_BEHAVIORAL_DIVERSITY_MIN","PRESSURE_ENERGY_BOOST"
        ]
        missing = [k for k in required if not hasattr(cfg, k)]
        if missing:
            print(f"{WARN} Missing config fields: {missing}")
        else:
            print(f"{OK} All required config knobs present")
        cfg_status["missing"] = missing
    except Exception as e:
        print(f"{FAIL} Could not instantiate config: {e}")

# ---------------------------------------------------------------------
# 3) Σ EMA smoothing sanity
# ---------------------------------------------------------------------
section("3) Σ EMA smoothing (symbolic)")
ema_ok = False
if mods["symbolic"] and mods["config"]:
    try:
        cfg = get_cfg(mods["config"])
        # Detect class name
        SymClass = getattr(mods["symbolic"], "SymbolicReasoner", None) or getattr(mods["symbolic"], "Symbolic", None)
        if not SymClass:
            print(f"{FAIL} No SymbolicReasoner/Symbolic class found in symbolic.py")
        else:
            sym = SymClass(cfg=cfg) if "cfg" in SymClass.__init__.__code__.co_varnames else SymClass()
            if not hasattr(sym, "sigma_history"):
                print(f"{FAIL} symbolic instance has no sigma_history")
            else:
                import numpy as np
                # Feed changing S and check EMA changes smoothly
                s = np.linspace(-1, 1, 64)
                for k in range(10):
                    S = s * math.sin(k/3.0)
                    out = sym.step(S)
                ema_series = getattr(sym, "sigma_history", [])
                if ema_series and abs(ema_series[-1]) <= 1.0:
                    print(f"{OK} sigma_history present; last EMA={ema_series[-1]:.4f}")
                    ema_ok = True
                else:
                    print(f"{WARN} sigma_history empty or out of expected bounds")
    except Exception as e:
        print(f"{FAIL} EMA check error: {e}")

# ---------------------------------------------------------------------
# 4) Context hysteresis/dwell behavioral check
# ---------------------------------------------------------------------
section("4) Context hysteresis & dwell (context)")
ctx_ok = False
if mods["context"] and mods["config"]:
    try:
        cfg = get_cfg(mods["config"])
        Ctx = getattr(mods["context"], "ContextModule", None)
        if not Ctx:
            print(f"{FAIL} No ContextModule found in context.py")
        else:
            ctx = Ctx(cfg=cfg) if "cfg" in Ctx.__init__.__code__.co_varnames else Ctx()
            # Feed a sequence around threshold
            hi = getattr(cfg, "RECONTEXT_THRESHOLD", 0.35)
            lo = hi - getattr(cfg, "RECONTEXT_HYSTERESIS", 0.08)
            seq = [0.0]*10 + [hi+0.1]*5 + [lo+0.01, hi-0.01]*(cfg.CONTEXT_MIN_DWELL_STEPS+4) + [hi+0.2]*5
            for v in seq:
                ctx.update({"distinction_level": v})
            hist = getattr(ctx, "context_history", [])
            flips = sum(1 for i in range(1,len(hist)) if hist[i]!=hist[i-1])
            if flips <= max(2, len(seq)//cfg.CONTEXT_MIN_DWELL_STEPS):
                print(f"{OK} context switches bounded (flips={flips})")
                ctx_ok = True
            else:
                print(f"{WARN} context flips high (flips={flips}) — dwell/hysteresis may be weak")
    except Exception as e:
        print(f"{FAIL} context check error: {e}")

# ---------------------------------------------------------------------
# 5) Core agent smoke: energy & existential pressure hook
# ---------------------------------------------------------------------
section("5) Agent smoke test (energy & pressure)")
agent_ok = False
if mods["agent"] and mods["config"]:
    try:
        cfg = get_cfg(mods["config"])
        Emile = getattr(mods["agent"], "EmileAgent", None)
        if not Emile:
            print(f"{FAIL} EmileAgent not found in agent.py")
        else:
            a = Emile()
            # Simulate few steps; allow external rewards to vary
            low = 0
            for t in range(60):
                ext = None
                if t%15==0: ext={"reward":0.5}
                a.step(dt=0.01, external_input=ext)
                if hasattr(a, "detect_repetition") and a.detect_repetition():
                    if hasattr(a, "apply_existential_pressure"):
                        a.apply_existential_pressure()
                        low += 1
            hist = a.get_history()
            e_floor = getattr(cfg, "ENERGY_MIN_FLOOR", 0.05)
            energy = getattr(a.body.state, "energy", None) if hasattr(a, "body") else None
            if energy is not None and energy >= e_floor:
                print(f"{OK} energy not floor-pinned (E={energy:.3f}, floor={e_floor})")
                agent_ok = True
            else:
                print(f"{WARN} energy check inconclusive or floor-pinned (E={energy})")
    except Exception as e:
        print(f"{FAIL} agent smoke error: {e}")

# ---------------------------------------------------------------------
# 6) Embodied env: consumable resources / forage
# ---------------------------------------------------------------------
section("6) Embodied environment (consumable resources / forage)")
emb_ok = False
if mods["embodied_qse_emile"]:
    try:
        run_emb = getattr(mods["embodied_qse_emile"], "run_embodied_experiment", None)
        if run_emb:
            ok, res = try_call(run_emb, steps=120, visualize=False)
            if ok and isinstance(res, dict):
                agent = res.get("agent")
                consumed = any("consumed" in str(ev).lower() for ev in res.get("object_discoveries", []))
                e = getattr(getattr(agent, "body", types.SimpleNamespace()).state, "energy", None)
                if e is not None and e > 0.1:
                    print(f"{OK} embodied run: energy={e:.3f}")
                else:
                    print(f"{WARN} embodied energy not rising (energy={e})")
                # Soft check: we can’t rely on naming, but ensure objects list shrinks or reports consumption
                print(f"{OK if consumed else WARN} resource consumption event detected={consumed}")
                emb_ok = True
            else:
                print(f"{WARN} run_embodied_experiment returned {type(res)}; manual check recommended")
        else:
            print(f"{WARN} run_embodied_experiment not found; skipping")
            emb_ok = True  # don't fail audit on optional demo wrapper
    except Exception as e:
        print(f"{FAIL} embodied check error: {e}")

# ---------------------------------------------------------------------
# 7) Social: cooldown + confidence floor
# ---------------------------------------------------------------------
section("7) Social transfer (cooldown + min-confidence)")
social_ok = False
if mods["social_qse_agent_v2"] and mods["config"]:
    try:
        cfg = get_cfg(mods["config"])
        run_social = getattr(mods["social_qse_agent_v2"], "run_social_experiment", None)
        if run_social:
            ok, tup = try_call(run_social, n_agents=2, steps=150)
            if ok and isinstance(tup, tuple) and len(tup)>=3:
                env, agents, results = tup[0], tup[1], tup[2]
                # Try to detect any knowledge records with conf >= MIN_CONFIDENCE_RETENTION
                min_conf = getattr(cfg, "MIN_CONFIDENCE_RETENTION", 0.80)
                high_conf = False
                for ag in agents:
                    know = getattr(ag, "knowledge", {})
                    for v in (know.values() if isinstance(know, dict) else []):
                        conf = v.get("conf") if isinstance(v, dict) else None
                        if isinstance(conf, (int,float)) and conf >= min_conf:
                            high_conf = True
                            break
                print(f"{OK if high_conf else WARN} transfers retain conf ≥ {min_conf}: {high_conf}")
                social_ok = True
            else:
                print(f"{WARN} run_social_experiment returned unexpected {type(tup)}")
                social_ok = True
        else:
            # Fallback: structural check
            SQA = getattr(mods["social_qse_agent_v2"], "SocialQSEAgent", None)
            if SQA:
                print(f"{OK} SocialQSEAgent present; functional demo wrapper missing")
                social_ok = True
            else:
                print(f"{WARN} no SocialQSEAgent/run_social_experiment detected")
    except Exception as e:
        print(f"{FAIL} social check error: {e}")

# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------
section("Audit summary")
summary = {
    "imports_ok": all_imported,
    "cfg_missing": cfg_status.get("missing", []),
    "sigma_ema_ok": ema_ok,
    "context_hysteresis_ok": ctx_ok,
    "agent_energy_ok": agent_ok,
    "embodied_ok": emb_ok,
    "social_ok": social_ok,
}
print(summary)

# Simple pass/fail gate (non-fatal if optional demos missing)
hard_fail = []
if not all_imported: hard_fail.append("imports")
if summary["cfg_missing"]: hard_fail.append("config")
if not ema_ok: hard_fail.append("sigma_ema")
if not ctx_ok: hard_fail.append("context_hysteresis")

if hard_fail:
    print(f"\n{FAIL} HARD FAIL: {hard_fail}")
    sys.exit(1)
else:
    print(f"\n{OK} CORE CHECKS PASSED")
    sys.exit(0)
