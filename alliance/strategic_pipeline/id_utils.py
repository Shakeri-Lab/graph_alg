"""Identifier normalization helpers for alliance-network artifacts.

SDC/Compustat CUSIPs are six-character issuer identifiers.  Pandas will
sometimes infer all-digit CUSIPs as integers when reading CSVs, which drops
leading zeros and splits the same firm into multiple identities.  All pipeline
code should route CUSIP-like fields through these helpers before grouping.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd


_MISSING = {"", "nan", "none", "null", "<na>", "na"}


def normalize_cusip(value) -> str | None:
    """Return a canonical CUSIP-6 string, or ``None`` for missing values."""
    if value is None or pd.isna(value):
        return None
    s = str(value).strip().upper()
    if s.lower() in _MISSING:
        return None
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    s = s.replace(" ", "")
    if len(s) < 6 and s.isdigit():
        s = s.zfill(6)
    return s[:6] if len(s) > 6 else s


def normalize_cusip_series(series: pd.Series) -> pd.Series:
    """Normalize a pandas Series containing CUSIP-like values."""
    return series.map(normalize_cusip).astype("string")


def normalize_cusip_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Return ``df`` with selected CUSIP columns normalized in-place."""
    for col in columns:
        if col in df.columns:
            df[col] = normalize_cusip_series(df[col])
    return df
