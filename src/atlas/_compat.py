"""Small compatibility helpers for different pandas and NumPy versions.

pandas 2.2 renamed the month-end and quarter-end frequency aliases from "M"/"Q" to "ME"/"QE".
NumPy 2 renamed trapz to trapezoid. These constants let the code run on either version.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_PD_VERSION = tuple(int(x) for x in pd.__version__.split(".")[:2])
MONTH_END = "ME" if _PD_VERSION >= (2, 2) else "M"
QUARTER_END = "QE" if _PD_VERSION >= (2, 2) else "Q"
trapezoid = getattr(np, "trapezoid", None) or np.trapz
