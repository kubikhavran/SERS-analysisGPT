from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class Spectrum:
    """Jedno spektrum – datový model (UI-agnostický)."""
    id: str
    filename: str
    x: np.ndarray
    y: np.ndarray


def _read_two_columns(file_bytes: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """
    Načte 2 sloupce z TXT.
    Zkouší: whitespace/tab, čárku, středník.
    """
    bio = io.BytesIO(file_bytes)

    # 1) whitespace / tab
    try:
        df = pd.read_csv(bio, sep=r"\s+", header=None, engine="python")
    except Exception:
        df = None

    # 2) čárka
    if df is None or df.shape[1] < 2:
        bio.seek(0)
        try:
            df = pd.read_csv(bio, sep=",", header=None, engine="python")
        except Exception:
            df = None

    # 3) středník
    if df is None or df.shape[1] < 2:
        bio.seek(0)
        df = pd.read_csv(bio, sep=";", header=None, engine="python")

    if df.shape[1] < 2:
        raise ValueError("Soubor nemá alespoň 2 sloupce")

    df = df.iloc[:, :2]
    df.columns = ["x", "y"]

    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna()

    if len(df) == 0:
        raise ValueError("Soubor neobsahuje platná numerická data")

    df = df.sort_values(by="x")
    return df["x"].to_numpy(), df["y"].to_numpy()


def spectrum_from_upload(filename: str, file_bytes: bytes) -> Spectrum:
    x, y = _read_two_columns(file_bytes)
    return Spectrum(
        id=Path(filename).name,
        filename=Path(filename).name,
        x=x,
        y=y,
    )
