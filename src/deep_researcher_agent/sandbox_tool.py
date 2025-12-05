"""Lightweight Pyodide-style sandbox tool for safe-ish Python snippets."""

from __future__ import annotations

import contextlib
import io
import math
import statistics
from typing import Any, Dict, Optional

from langchain_core.tools import tool


@tool(
    "pyodide_sandbox",
    description=(
        "Execute a small Python snippet with limited builtins/math/statistics. "
        "Use this for numeric analysis like CVD/POC/footprint calculations. "
        "Data is passed in via the `data` variable. Assign to `result` to return a value."
    ),
)
def pyodide_sandbox(code: str, data: Any | None = None) -> Dict[str, Any]:
    """
    Execute small Python snippets in a constrained sandbox.

    - Exposes: math, statistics, and a curated set of safe builtins.
    - The variable `data` is available to the code and can contain lists/dicts of numeric data.
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
        "sorted": sorted,
    }

    safe_globals = {
        "__builtins__": allowed_builtins,
        "math": math,
        "statistics": statistics,
        "data": data,
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

__all__ = ["pyodide_sandbox"]
