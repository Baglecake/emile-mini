#!/usr/bin/env python3
"""
Entry point for emile-mini package when run as `python -m emile_mini`
"""

if __name__ == "__main__":
    import sys
    from .cli import main
    sys.exit(main())
