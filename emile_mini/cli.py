
# emile_mini/cli.py
import argparse, sys, inspect, importlib
import numpy as np
from . import __version__
from .social_qse_agent_v2 import run_social_experiment


def _call_with_known_kwargs(fn, **proposed):
    sig = inspect.signature(fn)
    allowed = {k: v for k, v in proposed.items() if k in sig.parameters}
    return fn(**allowed) if allowed else fn()


def build_parser():
    p = argparse.ArgumentParser(prog="emile-mini", description="émile-mini demos")
    p.add_argument("-V", "--version", action="version", version=f"emile-mini {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    # social
    social = sub.add_parser("social", help="Run the social demo")
    social.add_argument("--agents", type=int, default=3)
    social.add_argument("--steps", type=int, default=120)
    social.add_argument("--cluster-radius", type=int, default=4)
    social.set_defaults(func=cmd_social)

    # maze
    maze = sub.add_parser("maze", help="Visual maze demo (plots)")
    maze.add_argument("--steps", type=int, default=200)
    maze.add_argument("--size", type=int, default=12)
    maze.set_defaults(func=cmd_maze)

    # extinction
    ext = sub.add_parser("extinction", help="Learning → extinction → recovery")
    ext.add_argument("--episodes", type=int, default=1)
    ext.add_argument("--phase-steps", type=int, default=120)
    ext.set_defaults(func=cmd_extinction)

    # nav-demo
    nav = sub.add_parser("nav-demo", help="Run the memory-guided navigation demo")
    nav.add_argument("--size", type=int, default=20, help="Grid size (default: 20)")
    nav.add_argument("--max-steps", type=int, default=50, help="Max steps to run (default: 50)")
    nav.add_argument("--quadrant", choices=["NE", "NW", "SE", "SW", "C"], default="NE",
                     help="Target quadrant (default: NE)")
    nav.add_argument("--seed", type=int, default=42, help="RNG seed for deterministic runs")
    nav.add_argument("--verbose", action="store_true", help="Print per-step logs")
    nav.set_defaults(func=cmd_nav_demo)

    # battery
    bat = sub.add_parser("battery", help="Run the Cognitive Battery (A, C1, C2, optional PPO)")
    bat.add_argument("--episodes-a", type=int, default=5)
    bat.add_argument("--episodes-c1", type=int, default=3)
    bat.add_argument("--phases-c1", type=int, default=3)
    bat.add_argument("--steps-per-phase", type=int, default=150)
    bat.add_argument("--episodes-c2", type=int, default=5)
    bat.add_argument("--max-steps", type=int, default=200)
    bat.add_argument("--seed", type=int, default=42)
    bat.add_argument("--prefix", type=str, default="emile_cognitive_battery")
    bat.add_argument("--ppo", action="store_true")
    bat.add_argument("--ppo-timesteps", type=int, default=50000)
    bat.add_argument("--size", type=int, default=20)
    bat.add_argument("--max-steps-nav", type=int, default=80)
    bat.add_argument("--obstacles", type=float, default=0.15)
    bat.add_argument("--start-mode", choices=["random","center"], default="random")
    bat.add_argument("--quadrant", choices=["NE","NW","SE","SW","C"], default="NE")
    bat.set_defaults(func=cmd_battery)

        # nav-ppo-train
    ppo_tr = sub.add_parser("nav-ppo-train", help="Train PPO on the NavEnv and save a model")
    ppo_tr.add_argument("--timesteps", type=int, default=50000)
    ppo_tr.add_argument("--size", type=int, default=20)
    ppo_tr.add_argument("--max-steps", type=int, default=80)
    ppo_tr.add_argument("--obstacles", type=float, default=0.15, help="Obstacle density 0..1")
    ppo_tr.add_argument("--quadrant", choices=["NE","NW","SE","SW","C"], default="NE")
    ppo_tr.add_argument("--start-mode", choices=["random","center"], default="random")
    ppo_tr.add_argument("--seed", type=int, default=42)
    ppo_tr.add_argument("--model", type=str, default="ppo_nav.zip")

    # reward/shaping knobs (these match your NavEnv __init__ defaults)
    ppo_tr.add_argument("--progress-k", type=float, default=0.5)
    ppo_tr.add_argument("--step-cost", type=float, default=0.02)
    ppo_tr.add_argument("--turn-penalty", type=float, default=0.01)
    ppo_tr.add_argument("--collision-penalty", type=float, default=0.10)
    ppo_tr.add_argument("--success-bonus", type=float, default=2.0)

    ppo_tr.set_defaults(func=cmd_nav_ppo_train)

    # nav-compare
    cmpc = sub.add_parser("nav-compare", help="Compare PPO vs handcrafted (nav agent) on NavEnv")
    cmpc.add_argument("--episodes", type=int, default=200)
    cmpc.add_argument("--timesteps", type=int, default=50000, help="Train timesteps if model missing")
    cmpc.add_argument("--size", type=int, default=20)
    cmpc.add_argument("--max-steps", type=int, default=80)
    cmpc.add_argument("--obstacles", type=float, default=0.15)
    cmpc.add_argument("--quadrant", choices=["NE","NW","SE","SW","C"], default="NE")
    cmpc.add_argument("--start-mode", choices=["random","center"], default="random")
    cmpc.add_argument("--seed", type=int, default=42)
    cmpc.add_argument("--model", type=str, default="ppo_nav.zip")

    # same reward/shaping knobs
    cmpc.add_argument("--progress-k", type=float, default=0.5)
    cmpc.add_argument("--step-cost", type=float, default=0.02)
    cmpc.add_argument("--turn-penalty", type=float, default=0.01)
    cmpc.add_argument("--collision-penalty", type=float, default=0.10)
    cmpc.add_argument("--success-bonus", type=float, default=2.0)

    cmpc.set_defaults(func=cmd_nav_compare)

        # nav-report
    rep = sub.add_parser("nav-report", help="Run quadrant & density sweeps and save CSV/JSON")
    rep.add_argument("--episodes", type=int, default=400, help="Episodes per condition")
    rep.add_argument("--timesteps", type=int, default=50000, help="Train timesteps if PPO model missing")
    rep.add_argument("--size", type=int, default=20)
    rep.add_argument("--max-steps", type=int, default=80)
    rep.add_argument("--start-mode", choices=["random","center"], default="random")
    rep.add_argument("--seed", type=int, default=123)
    rep.add_argument("--model", type=str, default="ppo_nav.zip")

    # quadrant sweep controls
    rep.add_argument("--quadrants", type=str, default="NE,NW,SE,SW,C",
                     help="Comma-separated list for quadrant sweep")
    rep.add_argument("--obstacles", type=float, default=0.20,
                     help="Obstacle density for quadrant sweep")

    # density sweep controls
    rep.add_argument("--densities", type=str, default="0.10,0.20,0.30",
                     help="Comma-separated list for density sweep")
    rep.add_argument("--density-quadrant", choices=["NE","NW","SE","SW","C"], default="NE",
                     help="Fixed quadrant for density sweep")

    # reward shaping knobs (forwarded to NavEnv)
    rep.add_argument("--progress-k", type=float, default=0.5)
    rep.add_argument("--step-cost", type=float, default=0.02)
    rep.add_argument("--turn-penalty", type=float, default=0.01)
    rep.add_argument("--collision-penalty", type=float, default=0.10)
    rep.add_argument("--success-bonus", type=float, default=2.0)

    rep.add_argument("--prefix", type=str, default="nav_report",
                     help="Output file prefix (CSV/JSON)")

    rep.set_defaults(func=cmd_nav_report)

    return p


def cmd_social(args):
    run_social_experiment(
        n_agents=args.agents,
        steps=args.steps,
        cluster_spawn=True,
        cluster_radius=args.cluster_radius,
    )


def cmd_maze(args):
    try:
        from .visual_maze_demo import main as maze_main
    except Exception as e:
        print(f"[maze] import failed: {e}", file=sys.stderr)
        return 2
    return _call_with_known_kwargs(maze_main, steps=args.steps, size=args.size)


def cmd_extinction(args):
    try:
        from .extinction_experiment import main as extinction_main
    except Exception as e:
        print(f"[extinction] import failed: {e}", file=sys.stderr)
        return 2
    return _call_with_known_kwargs(extinction_main, episodes=args.episodes, phase_steps=args.phase_steps)

def cmd_nav_report(args):
    try:
        from .nav_report import run_report
    except Exception as e:
        print(f"[nav-report] import failed: {e}", file=sys.stderr)
        return 2

    # parse comma-separated lists
    quadrants = [s.strip() for s in args.quadrants.split(",") if s.strip()]
    densities = [float(s.strip()) for s in args.densities.split(",") if s.strip()]

    try:
        paths = run_report(
            quadrants=quadrants,
            obstacles=args.obstacles,
            densities=densities,
            density_quadrant=args.density_quadrant,
            episodes=args.episodes,
            timesteps=args.timesteps,
            size=args.size,
            max_steps=args.max_steps,
            start_mode=args.start_mode,
            seed=args.seed,
            model=args.model,
            progress_k=args.progress_k,
            step_cost=args.step_cost,
            turn_penalty=args.turn_penalty,
            collision_penalty=args.collision_penalty,
            success_bonus=args.success_bonus,
            prefix=args.prefix,
        )
        print(f"[nav-report] done. CSV: {paths['csv']} JSON: {paths['json']}")
        return 0
    except Exception as e:
        print(f"[nav-report] failed: {e}", file=sys.stderr)
        return 2



# ---------- Navigation demo command (package-resident) ----------

def _import_nav_module():
    """
    Import the nav demo inside the package.
    Tries:
      - emile_mini.complete_navigation_system_d
      - emile_mini.complete_navigation_system_b (fallback)
      - relative imports (for python -m)
    """
    candidates = [
        "emile_mini.complete_navigation_system_d",
        "emile_mini.complete_navigation_system_b",
        ".complete_navigation_system_d",
        ".complete_navigation_system_b",
    ]
    last_err = None
    for name in candidates:
        try:
            if name.startswith("."):
                return importlib.import_module(name, package=__package__)
            return importlib.import_module(name)
        except Exception as e:
            last_err = e
            continue
    raise ImportError(
        "Could not import emile_mini.complete_navigation_system_d or _b "
        f"(last error: {last_err})"
    )


def _run_nav_demo_inline(size=20, max_steps=50, quadrant="NE", seed=42, verbose=False):
    """
    Inline runner using the module classes so CLI options work.
    """
    nav = _import_nav_module()

    ClearPathEnvironment = getattr(nav, "ClearPathEnvironment")
    ProactiveEmbodiedQSEAgent = getattr(nav, "ProactiveEmbodiedQSEAgent")

    from emile_mini.config import QSEConfig
    cfg = QSEConfig()

    np.random.seed(seed)
    print("🧪 TESTING ENHANCED NAVIGATION SYSTEM")
    print("=" * 50)

    env = ClearPathEnvironment(size=size)
    agent = ProactiveEmbodiedQSEAgent(config=cfg)

    # Start centered; heading east
    agent.body.state.position = (size // 2, size // 2)
    agent.body.state.orientation = 0.0

    cue = {
        "type": "navigation_cue",
        "target_quadrant": quadrant,
        "instruction": "Navigate to target quadrant",
        "priority": "high"
    }
    agent.receive_memory_cue(cue)

    print(f"📍 Start: {agent.body.state.position}")
    print(f"🎯 Target: {agent.memory_goal.target_position}")
    print(f"🧭 Orientation: {agent.body.state.orientation:.3f} rad")

    init_dist = float(np.linalg.norm(
        np.array(agent.memory_goal.target_position) -
        np.array(agent.body.state.position)
    ))
    print(f"📏 Initial distance: {init_dist:.1f}")

    success = False
    distances = [init_dist]
    cue_rates = [0.0]
    actions = []

    for step in range(max_steps):
        result = agent.embodied_step(env)

        distances.append(result["target_distance"])
        cue_rates.append(result["cue_follow_rate"])
        actions.append(result["action"])

        if verbose or step % 5 == 0 or step < 10:
            print(
                f"  Step {step:2d}: pos={result['new_position']}, "
                f"dist={result['target_distance']:.1f}, "
                f"cue_rate={result['cue_follow_rate']:.3f}, "
                f"action={result['action']} (bias: {result['action_bias']:.2f})"
            )

        if result["action"] == "SUCCESS" or result["target_distance"] < 2.0:
            print(f"🎯 SUCCESS: Reached target at step {step}!")
            success = True
            break

    final_dist = distances[-1]
    recent = cue_rates[-10:]
    active_recent = [r for r in recent if r > 0.1]
    avg_cue_rate = float(np.mean(active_recent) if active_recent else np.mean(recent))
    dist_improvement = init_dist - final_dist

    print("\n📊 ENHANCED NAVIGATION RESULTS:")
    print(f"Initial Distance: {init_dist:.1f}")
    print(f"Final Distance: {final_dist:.1f}")
    print(f"Distance Improvement: {dist_improvement:+.1f}")
    print(f"Average Cue Follow Rate (last 10): {avg_cue_rate:.3f}")
    print(f"Peak Cue Follow Rate: {max(cue_rates):.3f}")
    print(f"Success: {success}")

    moved = len(set(distances)) > 1
    progressed = dist_improvement > 3.0
    followed_cues = (avg_cue_rate > 0.35) or (max(cue_rates) > 0.7)

    print(f"Agent Movement: {'✅ YES' if moved else '❌ NO'}")
    print(f"Distance Progress: {'✅ YES' if progressed else '❌ NO'}")
    print(f"Cue Following: {'✅ YES' if followed_cues else '❌ NO'}")

    if success and followed_cues:
        print("\n🎉 EXCELLENT: Navigation system working perfectly!")
        return 0
    elif success and progressed:
        print("\n🎉 EXCELLENT: Navigation successful with good progress!")
        return 0
    elif progressed and moved:
        print("\n🟡 GOOD: Significant progress made")
        return 0
    else:
        print("\n❌ NEEDS WORK: Navigation not working")
        return 1

def cmd_battery(args):
    try:
        from .cognitive_battery import main as battery_main
    except Exception as e:
        print(f"[battery] import failed: {e}", file=sys.stderr)
        return 2
    return battery_main(
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
    )


def cmd_nav_demo(args):
    """
    Force the inline runner so CLI options are ALWAYS respected.
    No fallback to test_complete_navigation(); no guessing.
    """
    try:
        _ = _import_nav_module()  # ensure module exists inside the package
    except Exception as e:
        print(f"[nav-demo] import failed: {e}", file=sys.stderr)
        return 2

    # If you want to see the parsed args, run with --verbose
    if args.verbose:
        print(f"[nav-demo] args -> size={args.size}, max_steps={args.max_steps}, "
              f"quadrant={args.quadrant}, seed={args.seed}, verbose={args.verbose}")

    try:
        return _run_nav_demo_inline(
            size=args.size,
            max_steps=args.max_steps,
            quadrant=args.quadrant,
            seed=args.seed,
            verbose=args.verbose
        )
    except Exception as e:
        print(f"[nav-demo] inline run failed: {e}", file=sys.stderr)
        return 2

def cmd_nav_ppo_train(args):
    try:
        from .ppo_nav_baseline import train_ppo, make_env_kwargs
    except Exception as e:
        print(f"[nav-ppo-train] import failed: {e}", file=sys.stderr)
        return 2

    try:
        kw = make_env_kwargs(
            size=args.size,
            max_steps=args.max_steps,
            obstacle_density=args.obstacles,
            quadrant=args.quadrant,
            start_mode=args.start_mode,
            seed=args.seed,
            progress_k=args.progress_k,
            step_cost=args.step_cost,
            turn_penalty=args.turn_penalty,
            collision_penalty=args.collision_penalty,
            success_bonus=args.success_bonus,
        )
        path = train_ppo(timesteps=args.timesteps, save_path=args.model, **kw)
        print(f"[nav-ppo-train] saved: {path}")
        return 0
    except Exception as e:
        print(f"[nav-ppo-train] failed: {e}", file=sys.stderr)
        return 2


def cmd_nav_compare(args):
    try:
        from .ppo_nav_baseline import compare_ppo_vs_handcrafted, make_env_kwargs
    except Exception as e:
        print(f"[nav-compare] import failed: {e}", file=sys.stderr)
        return 2

    try:
        kw = make_env_kwargs(
            size=args.size,
            max_steps=args.max_steps,
            obstacle_density=args.obstacles,
            quadrant=args.quadrant,
            start_mode=args.start_mode,
            seed=args.seed,
            progress_k=args.progress_k,
            step_cost=args.step_cost,
            turn_penalty=args.turn_penalty,
            collision_penalty=args.collision_penalty,
            success_bonus=args.success_bonus,
        )
        stats = compare_ppo_vs_handcrafted(
            n_episodes=args.episodes,
            timesteps=args.timesteps,
            model_path=args.model,
            **kw
        )
        print("\n=== RESULTS ===")
        for k in ("ppo", "handcrafted"):
            s = stats[k]
            print(f"{k.upper():>12}: success={s['success_rate']:.3f}  "
                  f"ret={s['avg_return']:.3f}  steps={s['avg_steps']:.1f}  SPL={s['avg_spl']:.3f}")
        return 0
    except Exception as e:
        print(f"[nav-compare] failed: {e}", file=sys.stderr)
        return 2



def main(argv=None):
    p = build_parser()
    args = p.parse_args(argv)
    try:
        rc = args.func(args)
        return 0 if rc is None else int(rc)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
