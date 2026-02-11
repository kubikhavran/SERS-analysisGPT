from __future__ import annotations

import numpy as np


def normalize_spectrum(
    y: np.ndarray,
    x: np.ndarray | None = None,
    method: str = "max",
) -> np.ndarray:
    """
    Normalizace spektra:
      - max: maximum = 1
      - area: plocha = 1 (integrál |y| dx)
      - minmax: 0–1
    """
    y = np.asarray(y)

    if method == "max":
        m = float(np.max(y))
        return y / m if m != 0 else y

    if method == "area":
        ay = np.abs(y)
        if x is None:
            area = float(np.sum((ay[:-1] + ay[1:]) * 0.5))
        else:
            x = np.asarray(x)
            dx = np.diff(x)
            if len(dx) != len(ay) - 1:
                area = float(np.sum((ay[:-1] + ay[1:]) * 0.5))
            else:
                area = float(np.sum((ay[:-1] + ay[1:]) * 0.5 * dx))
        return y / area if area != 0 else y

    if method == "minmax":
        mn = float(np.min(y))
        mx = float(np.max(y))
        return (y - mn) / (mx - mn) if (mx - mn) != 0 else y

    return y
