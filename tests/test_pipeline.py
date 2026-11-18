"""Section 10: integration."""

import pytest

from atlas.pipeline.dag import DAILY_SCHEDULE, Job, run, topological_order


def test_schedule_order_respects_dependencies():
    order = topological_order(DAILY_SCHEDULE)
    pos = {n: i for i, n in enumerate(order)}
    for j in DAILY_SCHEDULE:
        for n in j.needs:
            assert pos[n] < pos[j.name]
    assert order.index("optimiser") > order.index("risk") > order.index("forecast")


def test_failed_job_skips_dependants():
    def boom():
        raise RuntimeError("feed down")
    jobs = [Job("a", "", "0", func=boom), Job("b", "", "1", ["a"]), Job("c", "", "2")]
    st = run(jobs, retries=0, wait_seconds=0)
    assert st == {"a": "failed", "c": "ok", "b": "skipped"}


@pytest.mark.slow
def test_demo_runs_end_to_end(tmp_path):
    from atlas.pipeline.demo import run_demo
    res = run_demo(n_stocks=60, n_days=1260, seed=11, out_dir=tmp_path, verbose=False, fast=True)
    assert (tmp_path / "forecasts.csv").exists() and (tmp_path / "risk.csv").exists()
    f = res["forecasts"]
    s = f[["p_bull", "p_base", "p_bear"]].sum(axis=1)
    assert ((s - 1).abs() < 1e-9).all()
    assert res["gross"] <= 3.0 + 1e-6 and abs(res["net"]) <= 0.05 + 1e-6
    assert not res["signal_tests"].loc["noise", "bh_survives"]
