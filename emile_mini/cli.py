
# emile_mini/cli.py
import argparse, sys, inspect
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
    # Try to pass steps/size if supported; otherwise just run.
    return _call_with_known_kwargs(maze_main, steps=args.steps, size=args.size)

def cmd_extinction(args):
    try:
        from .extinction_experiment import main as extinction_main
    except Exception as e:
        print(f"[extinction] import failed: {e}", file=sys.stderr)
        return 2
    return _call_with_known_kwargs(extinction_main, episodes=args.episodes, phase_steps=args.phase_steps)

def main(argv=None):
    p = build_parser()
    args = p.parse_args(argv)
    try:
        rc = args.func(args)
        return 0 if rc is None else int(rc)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130

if __name__ == "__main__":
    raise SystemExit(main())
