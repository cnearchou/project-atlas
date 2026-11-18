"""Production gate (Spec §9.4).

A component goes live only if it passes every check:
1. Newey–West t-statistic above 3.
2. Survives the Benjamini–Hochberg correction across all signals tested.
3. Adds value in Fama–MacBeth regressions beyond known factors (|t| > 3 on the new signal).
4. Holds out-of-sample, across regions and across macro regimes (same sign of mean IC in every split).
5. Robust to parameter changes (all neighbouring variants have the same sign and t > 2).
6. Profitable after transaction costs (net long–short mean > 0).
7. Deflated Sharpe ratio above 0.95.
Checks whose inputs were not supplied are reported as 'not run', and a component with any check
not run cannot pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GateResult:
    checks: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(v is True for v in self.checks.values())

    def report(self) -> str:
        lines = [f"  {k:<28} {'PASS' if v is True else ('not run' if v is None else 'FAIL')}" for k, v in self.checks.items()]
        return "\n".join(lines + [f"  {'RESULT':<28} {'PASSED' if self.passed else 'NOT PASSED'}"])


def production_gate(t_nw: float | None = None, bh_survives: bool | None = None, fm_t: float | None = None,
                    split_ics: list[float] | None = None, variant_ts: list[float] | None = None,
                    net_mean: float | None = None, dsr: float | None = None,
                    t_min: float = 3.0, dsr_min: float = 0.95) -> GateResult:
    c = {}
    c["newey_west_t > 3"] = None if t_nw is None else bool(t_nw > t_min)
    c["benjamini_hochberg"] = None if bh_survives is None else bool(bh_survives)
    c["fama_macbeth"] = None if fm_t is None else bool(abs(fm_t) > t_min)
    c["out_of_sample_and_regimes"] = None if split_ics is None else bool(len(set(np.sign(split_ics))) == 1 and np.sign(split_ics[0]) != 0)
    c["parameter_robustness"] = None if variant_ts is None else bool(all(abs(t) > 2 for t in variant_ts) and len(set(np.sign(variant_ts))) == 1)
    c["after_costs"] = None if net_mean is None else bool(net_mean > 0)
    c["deflated_sharpe > 0.95"] = None if dsr is None else bool(dsr > dsr_min)
    return GateResult(c)
