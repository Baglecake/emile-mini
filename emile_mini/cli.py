# emile_mini/cli.py
import argparse
import sys

from . import __version__
from .social_qse_agent_v2 import run_social_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="emile-mini",
        description="Émile-mini demos and utilities",
    )
    parser.add_argument(
        "-V", "--version",
        action="version",
        version=f"emile-mini {__version__}"
    )

    sub = parser.add_subparsers(dest="cmd")

    social = sub.add_parser("social", help="Run the social demo")
    social.add_argument("--agents", type=int, default=3, help="Number of agents")
    social.add_argument("--steps", type=int, default=120, help="Number of steps")
    social.add_argument("--cluster-radius", type=int, default=4, help="Initial cluster radius")
    social.set_defaults(func=cmd_social)

    return parser


def cmd_social(args: argparse.Namespace) -> None:
    run_social_experiment(
        n_agents=args.agents,
        steps=args.steps,
        cluster_spawn=True,
        cluster_radius=args.cluster_radius,
    )


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    try:
        args.func(args)
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
