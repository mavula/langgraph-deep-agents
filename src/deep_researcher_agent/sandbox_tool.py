"""Lightweight Pyodide-style sandbox tool for safe-ish Python snippets."""

from __future__ import annotations

import contextlib
import io
import json
import math
import statistics
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from langchain_core.tools import tool


def _basic_sort_by_key(items: Any, key: str) -> Any:
    """Manual insertion sort that avoids relying on built-in `sorted` or imports."""

    if not isinstance(items, list):
        return items

    sorted_items = list(items)
    for i in range(1, len(sorted_items)):
        current = sorted_items[i]
        j = i - 1
        while j >= 0 and str(sorted_items[j].get(key, "")) > str(current.get(key, "")):
            sorted_items[j + 1] = sorted_items[j]
            j -= 1
        sorted_items[j + 1] = current
    return sorted_items


@tool(
    "pyodide_sandbox",
    description=(
        "Execute a small Python snippet with limited builtins/math/statistics plus json; imports are not available. "
        "Use only basic arithmetic/loops. Data is passed in via the `data` variable. "
        "Assign to `result` to return a value. Helpers: `basic_sort_by_key(items, key)` for manual sorting and `json` for simple parsing/serialization."
    ),
)
def pyodide_sandbox(code: str, data: Any | None = None) -> Dict[str, Any]:
    """
    Execute small Python snippets in a constrained sandbox.

    - Exposes: math, statistics, and a curated set of safe builtins (no imports).
    - The variable `data` is available to the code and can contain lists/dicts of numeric data.
    - `json` module is available for simple parsing/serialization.
    - To return a value, assign to `result` in the snippet.
    """

    allowed_builtins = {
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "round": round,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
    }

    safe_globals = {
        "__builtins__": allowed_builtins,
        "math": math,
        "statistics": statistics,
        "data": data,
        "basic_sort_by_key": _basic_sort_by_key,
        "json": json,
    }
    safe_locals: Dict[str, Any] = {}

    stdout_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_buffer):
            exec(code, safe_globals, safe_locals)
    except Exception as exc:  # noqa: BLE001
        return {"stdout": stdout_buffer.getvalue() or None, "result": None, "error": str(exc)}

    result = safe_locals.get("result")
    return {
        "stdout": stdout_buffer.getvalue() or None,
        "result": result,
        "error": None,
    }


def find_local_maxima(series: pd.Series, order: int = 3) -> np.ndarray:
    """Return indices of points that are maxima within a +/- ``order`` window."""

    x = series.to_numpy()
    n = len(x)
    peaks: list[int] = []
    for i in range(order, n - order):
        window = x[i - order : i + order + 1]
        if (
            np.isfinite(x[i])
            and x[i] == np.max(window)
            and x[i] != x[i - 1]
            and x[i] != x[i + 1]
        ):
            peaks.append(i)
    return np.array(peaks, dtype=int)


def detect_double_tops(
    df: pd.DataFrame,
    price_col: str = "close",
    order: int = 4,
    max_peak_gap: int = 60,
    peak_tolerance: float = 0.01,
    min_pullback: float = 0.03,
    confirm_breakdown: bool = True,
    breakdown_lookahead: int = 20,
    breakdown_buffer: float = 0.001,
) -> pd.DataFrame:
    """Detect double-top candidates and return their metadata as a DataFrame."""

    s = df[price_col].astype(float)
    peaks = find_local_maxima(s, order=order)
    results: list[dict[str, Any]] = []
    x = s.to_numpy()

    for a, b in zip(peaks[:-1], peaks[1:]):
        if b - a < 2 or (b - a) > max_peak_gap:
            continue

        p1, p2 = x[a], x[b]
        if not (np.isfinite(p1) and np.isfinite(p2)):
            continue

        peak_diff = abs(p2 - p1) / max(p1, 1e-12)
        if peak_diff > peak_tolerance:
            continue

        valley_idx = a + int(np.argmin(x[a : b + 1]))
        neckline = x[valley_idx]

        pullback1 = (p1 - neckline) / max(p1, 1e-12)
        pullback2 = (p2 - neckline) / max(p2, 1e-12)
        if min(pullback1, pullback2) < min_pullback:
            continue

        breakdown_idx = None
        if confirm_breakdown:
            end = min(len(x), b + 1 + breakdown_lookahead)
            post = x[b + 1 : end]
            thresh = neckline * (1 - breakdown_buffer)
            hits = np.where(post < thresh)[0]
            if len(hits) == 0:
                continue
            breakdown_idx = (b + 1) + int(hits[0])

        results.append(
            {
                "peak1_pos": a,
                "peak2_pos": b,
                "valley_pos": valley_idx,
                "peak1": float(p1),
                "peak2": float(p2),
                "neckline": float(neckline),
                "peak_diff_pct": float(peak_diff * 100),
                "pullback_pct": float(min(pullback1, pullback2) * 100),
                "breakdown_pos": breakdown_idx,
            }
        )

    out = pd.DataFrame(results)
    if isinstance(df.index, pd.DatetimeIndex) and not out.empty:
        out["peak1_time"] = df.index[out["peak1_pos"]].to_numpy()
        out["peak2_time"] = df.index[out["peak2_pos"]].to_numpy()
        out["valley_time"] = df.index[out["valley_pos"]].to_numpy()
        if out["breakdown_pos"].notna().any():
            breakdown_times = out["breakdown_pos"].dropna().astype(int)
            out.loc[breakdown_times.index, "breakdown_time"] = df.index[breakdown_times].to_numpy()
    return out


def _serialize_double_top_rows(df: pd.DataFrame) -> list[Dict[str, Any]]:
    """Convert a double-top results frame into JSON-serializable dictionaries."""

    def _convert(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, (np.datetime64,)):
            return pd.Timestamp(value).isoformat()
        if isinstance(value, np.generic):
            return value.item()
        return value

    records = []
    for row in df.to_dict(orient="records"):
        records.append({key: _convert(val) for key, val in row.items()})
    return records


@tool(
    "double_top_peaks",
    description=(
        "Analyze OHLC data for double-top structures. Provide candles with a price column (default `close`). "
        "Returns candidate peaks, neckline, and optional breakdown confirmation."
    ),
)
def double_top_peaks(
    candles: Any,
    price_col: str = "close",
    order: int = 4,
    max_peak_gap: int = 60,
    peak_tolerance: float = 0.01,
    min_pullback: float = 0.03,
    confirm_breakdown: bool = True,
    breakdown_lookahead: int = 20,
    breakdown_buffer: float = 0.001,
) -> Dict[str, Any]:
    """Detect swing-high double-top candidates from a candle sequence."""

    if isinstance(candles, pd.DataFrame):
        df = candles.copy()
    else:
        try:
            df = pd.DataFrame(candles)
        except ValueError as exc:  # pragma: no cover - defensive guard for malformed inputs
            raise ValueError("candles must be a DataFrame or list of records") from exc

    if df.empty:
        return {"double_tops": [], "count": 0, "error": "No candle data provided."}

    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found in data.")

    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    peaks = detect_double_tops(
        df,
        price_col=price_col,
        order=order,
        max_peak_gap=max_peak_gap,
        peak_tolerance=peak_tolerance,
        min_pullback=min_pullback,
        confirm_breakdown=confirm_breakdown,
        breakdown_lookahead=breakdown_lookahead,
        breakdown_buffer=breakdown_buffer,
    )

    if peaks.empty:
        return {"double_tops": [], "count": 0}

    records = _serialize_double_top_rows(peaks)
    return {"double_tops": records, "count": len(records)}


__all__ = ["pyodide_sandbox", "double_top_peaks", "detect_double_tops", "find_local_maxima"]
