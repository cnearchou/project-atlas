"""Daily processing schedule as a dependency graph (Spec §10.2).

Each job names the jobs whose outputs it needs. `topological_order` returns an order in which every
job runs after its inputs, which is what an orchestrator (Prefect or Dagster, Spec §10.3) does
automatically. The simple runner here executes registered callables in that order, retries failures
and records the status of each job, so the schedule can be tested without an orchestrator.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

log = logging.getLogger("atlas.pipeline")


@dataclass
class Job:
    name: str
    time_et: str
    section: str
    needs: list[str] = field(default_factory=list)
    func: Callable[[], object] | None = None


DAILY_SCHEDULE = [
    Job("news_ingest", "continuous", "3"),
    Job("macro_releases", "06:00", "2"),
    Job("market_close", "16:30", "0"),
    Job("factor_model", "17:00", "6", ["market_close"]),
    Job("flows", "17:15", "4", ["market_close"]),
    Job("behaviour", "17:30", "5", ["market_close", "factor_model"]),
    Job("fundamentals", "17:45", "1", ["market_close", "macro_releases"]),
    Job("regimes_and_betas", "18:00", "2", ["macro_releases", "factor_model"]),
    Job("forecast", "18:30", "7", ["factor_model", "flows", "behaviour", "fundamentals", "regimes_and_betas", "news_ingest"]),
    Job("risk", "19:00", "8", ["factor_model", "forecast"]),
    Job("optimiser", "19:30", "8", ["forecast", "risk"]),
    Job("monitoring", "19:45", "9", ["forecast", "risk", "optimiser"]),
]


def topological_order(jobs: list[Job]) -> list[str]:
    """Kahn's algorithm. Raises if the graph has a cycle or an unknown dependency."""
    names = {j.name for j in jobs}
    for j in jobs:
        missing = set(j.needs) - names
        if missing:
            raise ValueError(f"Job {j.name} needs unknown jobs {missing}")
    indeg = {j.name: len(j.needs) for j in jobs}
    children = {j.name: [] for j in jobs}
    for j in jobs:
        for n in j.needs:
            children[n].append(j.name)
    ready = [n for n, d in indeg.items() if d == 0]
    order = []
    while ready:
        n = ready.pop(0)
        order.append(n)
        for c in children[n]:
            indeg[c] -= 1
            if indeg[c] == 0:
                ready.append(c)
    if len(order) != len(jobs):
        raise ValueError("The schedule contains a cycle.")
    return order


def run(jobs: list[Job], retries: int = 2, wait_seconds: float = 1.0) -> dict[str, str]:
    """Run each job after its inputs; a job is skipped when any input failed."""
    by_name = {j.name: j for j in jobs}
    status: dict[str, str] = {}
    for name in topological_order(jobs):
        job = by_name[name]
        if any(status.get(n) != "ok" for n in job.needs):
            status[name] = "skipped"
            log.warning("Skipping %s because an input failed", name)
            continue
        if job.func is None:
            status[name] = "ok"
            continue
        for attempt in range(retries + 1):
            try:
                job.func()
                status[name] = "ok"
                break
            except Exception as e:  # noqa: BLE001 - every failure is logged and retried
                log.error("Job %s failed (attempt %d): %s", name, attempt + 1, e)
                status[name] = "failed"
                time.sleep(wait_seconds)
    return status
