"""The one interface through which every model family is trained and compared."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# A frame with ``title``, ``text`` and ``model_text`` -> calibrated P(unreliable), one per row.
Scorer = Callable[[pd.DataFrame], NDArray[np.float64]]

# ``(train, val, calib) -> Scorer``. ``val`` is for model selection and early stopping, ``calib``
# for calibration only; the frames also carry ``label``.
Fit = Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame], Scorer]
