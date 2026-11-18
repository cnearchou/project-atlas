"""Command-line interface.

    atlas demo [--stocks 200] [--days 1260] [--seed 7] [--fast]   run every layer on synthetic data
    atlas schedule                                                  print the daily job order (Spec §10.2)
    atlas register-hypotheses                                       pre-register H1–H3 (Section A.2)
"""

from __future__ import annotations

import argparse
import logging


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="atlas", description="Project Atlas command-line tools")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="run the end-to-end demo on synthetic data")
    d.add_argument("--stocks", type=int, default=200)
    d.add_argument("--days", type=int, default=1260)
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--fast", action="store_true", help="smaller simulations (for quick checks)")
    d.add_argument("--out", default=None, help="output folder (default: data/outputs)")
    sub.add_parser("schedule", help="print the daily processing schedule in dependency order")
    sub.add_parser("register-hypotheses", help="write H1–H3 to the research registry")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    if a.cmd == "demo":
        from atlas.pipeline.demo import run_demo
        run_demo(a.stocks, a.days, a.seed, a.out, verbose=True, fast=a.fast)
    elif a.cmd == "schedule":
        from atlas.pipeline.dag import DAILY_SCHEDULE, topological_order
        jobs = {j.name: j for j in DAILY_SCHEDULE}
        for name in topological_order(DAILY_SCHEDULE):
            j = jobs[name]
            print(f"{j.time_et:>10}  {name:<20} Section {j.section:<3} needs: {', '.join(j.needs) or '-'}")
    elif a.cmd == "register-hypotheses":
        from atlas.config import load_settings
        from atlas.research.hypotheses import register_all
        from atlas.research.registry import Registry
        ids = register_all(Registry(load_settings().path("registry")))
        print(f"Registered: {ids or 'already registered'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
