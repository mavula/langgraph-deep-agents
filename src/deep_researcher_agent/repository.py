"""Data-access helpers mirroring the TradingView MCP server."""

from __future__ import annotations

import datetime as dt
import decimal
import json
import os
import re
from typing import Any, Dict, List, Optional

from .database import DatabaseClient

ALLOWED_IDENTIFIER = re.compile(r"^[A-Za-z0-9_]+$")


def _sanitize_identifier(value: str, default: str) -> str:
    candidate = value or default
    if not ALLOWED_IDENTIFIER.match(candidate):
        raise ValueError(f"Invalid identifier provided: {candidate}")
    return candidate


def _normalize_time_frame(value: str) -> str:
    """Coerce incoming time frame values to match stored formats."""

    if value == "1":
        return "1m"
    elif value == "5":
        return "5m"
    elif value == "15":
        return "15m"
    elif value == "30":
        return "30m"
    elif value == "45":
        return "45m"
    elif value == "60":
        return "1H"

    return value


def _serialize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    serialized: Dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, dt.datetime):
            serialized[key] = value.isoformat()
        elif isinstance(value, dt.date):
            serialized[key] = value.isoformat()
        elif isinstance(value, decimal.Decimal):
            serialized[key] = float(value)
        else:
            serialized[key] = value
    return serialized


class CandleRepository:
    """Handles building SQL queries for tradingview_candle_data."""

    def __init__(
        self,
        db: DatabaseClient,
        table: str = "tradingview_candle_data",
        symbol_column: str = "symbol",
        time_frame_column: str = "time_frame",
        timestamp_column: str = "timestamp",
        exchange_column: Optional[str] = "exchange",
        footprint_table: str = "tradingview_volume_footprint",
        footprint_symbol_column: str = "symbol",
        footprint_time_frame_column: str = "time_frame",
        footprint_timestamp_column: str = "timestamp",
        footprint_exchange_column: Optional[str] = "exchange",
        cvd_table: str = "tradingview_candle_cvd",
        cvd_symbol_column: str = "symbol",
        cvd_time_frame_column: str = "time_frame",
        cvd_timestamp_column: str = "timestamp",
        cvd_exchange_column: Optional[str] = "exchange",
        ema_table: str = "tradingview_ema",
        ema_symbol_column: str = "symbol",
        ema_time_frame_column: str = "time_frame",
        ema_timestamp_column: str = "timestamp",
        ema_exchange_column: Optional[str] = "exchange",
        ema_20_column: str = "20_ema",
    ) -> None:
        self._db = db
        self._table = _sanitize_identifier(table, "tradingview_candle_data")
        self._symbol_column = _sanitize_identifier(symbol_column, "symbol")
        self._time_frame_column = _sanitize_identifier(time_frame_column, "time_frame")
        self._timestamp_column = _sanitize_identifier(timestamp_column, "timestamp")
        self._exchange_column = (
            _sanitize_identifier(exchange_column, "exchange") if exchange_column else None
        )
        self._footprint_table = _sanitize_identifier(footprint_table, "tradingview_volume_footprint")
        self._footprint_symbol_column = _sanitize_identifier(footprint_symbol_column, "symbol")
        self._footprint_time_frame_column = _sanitize_identifier(footprint_time_frame_column, "time_frame")
        self._footprint_timestamp_column = _sanitize_identifier(footprint_timestamp_column, "timestamp")
        self._footprint_exchange_column = (
            _sanitize_identifier(footprint_exchange_column, "exchange") if footprint_exchange_column else None
        )
        self._cvd_table = _sanitize_identifier(cvd_table, "tradingview_candle_cvd")
        self._cvd_symbol_column = _sanitize_identifier(cvd_symbol_column, "symbol")
        self._cvd_time_frame_column = _sanitize_identifier(cvd_time_frame_column, "time_frame")
        self._cvd_timestamp_column = _sanitize_identifier(cvd_timestamp_column, "timestamp")
        self._cvd_exchange_column = (
            _sanitize_identifier(cvd_exchange_column, "exchange") if cvd_exchange_column else None
        )
        self._ema_table = _sanitize_identifier(ema_table, "tradingview_ema")
        self._ema_symbol_column = _sanitize_identifier(ema_symbol_column, "symbol")
        self._ema_time_frame_column = _sanitize_identifier(ema_time_frame_column, "time_frame")
        self._ema_timestamp_column = _sanitize_identifier(ema_timestamp_column, "timestamp")
        self._ema_exchange_column = (
            _sanitize_identifier(ema_exchange_column, "exchange") if ema_exchange_column else None
        )
        self._ema_20_column = _sanitize_identifier(ema_20_column, "20_ema")

    @classmethod
    def from_env(cls, db: DatabaseClient) -> "CandleRepository":
        """Build repository using optional env overrides."""

        return cls(
            db=db,
            table=os.getenv("CANDLE_TABLE", "tradingview_candle_data"),
            symbol_column=os.getenv("CANDLE_SYMBOL_COLUMN", "symbol"),
            time_frame_column=os.getenv("CANDLE_TIME_FRAME_COLUMN", "time_frame"),
            timestamp_column=os.getenv("CANDLE_TIMESTAMP_COLUMN", "timestamp"),
            exchange_column=os.getenv("CANDLE_EXCHANGE_COLUMN", "exchange"),
            footprint_table=os.getenv("FOOTPRINT_TABLE", "tradingview_volume_footprint"),
            footprint_symbol_column=os.getenv("FOOTPRINT_SYMBOL_COLUMN", "symbol"),
            footprint_time_frame_column=os.getenv("FOOTPRINT_TIME_FRAME_COLUMN", "time_frame"),
            footprint_timestamp_column=os.getenv("FOOTPRINT_TIMESTAMP_COLUMN", "timestamp"),
            footprint_exchange_column=os.getenv("FOOTPRINT_EXCHANGE_COLUMN"),
            cvd_table=os.getenv("CVD_TABLE", "tradingview_candle_cvd"),
            cvd_symbol_column=os.getenv("CVD_SYMBOL_COLUMN", "symbol"),
            cvd_time_frame_column=os.getenv("CVD_TIME_FRAME_COLUMN", "time_frame"),
            cvd_timestamp_column=os.getenv("CVD_TIMESTAMP_COLUMN", "timestamp"),
            cvd_exchange_column=os.getenv("CVD_EXCHANGE_COLUMN", "exchange"),
            ema_table=os.getenv("EMA_TABLE", "tradingview_ema"),
            ema_symbol_column=os.getenv("EMA_SYMBOL_COLUMN", "symbol"),
            ema_time_frame_column=os.getenv("EMA_TIME_FRAME_COLUMN", "time_frame"),
            ema_timestamp_column=os.getenv("EMA_TIMESTAMP_COLUMN", "timestamp"),
            ema_exchange_column=os.getenv("EMA_EXCHANGE_COLUMN", "exchange"),
            ema_20_column=os.getenv("EMA_20_COLUMN", "20_ema"),
        )

    def fetch_candles(
        self,
        symbol: str,
        time_frame: str,
        limit: int,
        exchange: Optional[str] = None,
        start_timestamp: Optional[str] = None,
        end_timestamp: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return candle rows joined with footprint and CVD data when available."""

        limit = max(1, min(limit, 1000))
        time_frame = _normalize_time_frame(time_frame)

        select_columns = [
            f"c.{self._time_frame_column} AS time_frame",
            f"c.{self._symbol_column} AS symbol",
            f"c.{self._timestamp_column} AS timestamp",
            "c.open AS open",
            "c.high AS high",
            "c.low AS low",
            "c.close AS close",
            "c.volume AS volume",
            f"ema.{self._ema_20_column} AS {self._ema_20_column}",
            "fp.volume_delta AS volume_delta",
            "fp.poc AS poc",
            "fp.vah AS vah",
            "fp.val AS val",
            "fp.levels AS footprint_levels",
            "fp.total_fp_volume AS total_fp_volume",
            "fp.volume_diff AS volume_diff",
            "cvd.open AS cvd_open",
            "cvd.high AS cvd_high",
            "cvd.low AS cvd_low",
            "cvd.close AS cvd_close",
            "cvd.wick_color AS cvd_wick_color",
        ]

        sql_parts = [
            f"SELECT {', '.join(select_columns)}",
            f"FROM {self._table} AS c",
            f"JOIN {self._footprint_table} AS fp ON "
            f"fp.{self._footprint_symbol_column} = c.{self._symbol_column} "
            f"AND fp.{self._footprint_time_frame_column} = c.{self._time_frame_column} "
            f"AND fp.{self._footprint_timestamp_column} = c.{self._timestamp_column}",
        ]
        ema_join = (
            f"LEFT JOIN {self._ema_table} AS ema ON "
            f"ema.{self._ema_symbol_column} = c.{self._symbol_column} "
            f"AND ema.{self._ema_time_frame_column} = c.{self._time_frame_column} "
            f"AND ema.{self._ema_timestamp_column} = c.{self._timestamp_column}"
        )
        if self._exchange_column and self._ema_exchange_column:
            ema_join += f" AND ema.{self._ema_exchange_column} = c.{self._exchange_column}"
        sql_parts.extend(
            [
                ema_join,
                f"LEFT JOIN {self._cvd_table} AS cvd ON "
                f"cvd.{self._cvd_symbol_column} = c.{self._symbol_column} "
                f"AND cvd.{self._cvd_time_frame_column} = c.{self._time_frame_column} "
                f"AND cvd.{self._cvd_timestamp_column} = c.{self._timestamp_column}",
                "WHERE {symbol_col} = %s AND {time_frame_col} = %s".format(
                    symbol_col=f"c.{self._symbol_column}",
                    time_frame_col=f"c.{self._time_frame_column}",
                ),
            ]
        )
        params: List[Any] = [symbol, time_frame]

        if exchange is not None:
            if not self._exchange_column:
                raise ValueError("Exchange filtering requested but no exchange column configured.")
            sql_parts.append(f"AND c.{self._exchange_column} = %s")
            params.append(exchange)

        if start_timestamp is not None:
            sql_parts.append(f"AND c.{self._timestamp_column} >= %s")
            params.append(start_timestamp)
        if end_timestamp is not None:
            sql_parts.append(f"AND c.{self._timestamp_column} <= %s")
            params.append(end_timestamp)

        sql_parts.append(f"ORDER BY c.{self._timestamp_column} DESC")
        sql_parts.append("LIMIT %s")
        params.append(limit)

        rows = self._db.fetch_all(" ".join(sql_parts), params)
        rows.reverse()

        serialized_rows: List[Dict[str, Any]] = []
        for row in rows:
            serialized = _serialize_row(row)

            levels_value = serialized.get("footprint_levels")
            if isinstance(levels_value, (bytes, bytearray)):
                try:
                    levels_value = levels_value.decode("utf-8")
                    serialized["footprint_levels"] = levels_value
                except UnicodeDecodeError:
                    pass
            if isinstance(levels_value, str):
                try:
                    serialized["footprint_levels"] = json.loads(levels_value)
                except ValueError:
                    pass

            cvd_open = serialized.pop("cvd_open", None)
            cvd_high = serialized.pop("cvd_high", None)
            cvd_low = serialized.pop("cvd_low", None)
            cvd_close = serialized.pop("cvd_close", None)
            cvd_wick_color = serialized.pop("cvd_wick_color", None)
            cvd: Dict[str, Any] = {
                "open": cvd_open,
                "high": cvd_high,
                "low": cvd_low,
                "close": cvd_close,
            }
            wick_color_value: Optional[int] = None
            if cvd_wick_color is not None:
                wick_color_value = 1 if cvd_wick_color != 0 else 0
            elif cvd_close is not None and cvd_open is not None:
                wick_color_value = 1 if cvd_close < cvd_open else 0
            if wick_color_value is not None:
                cvd["wick_color"] = "red" if wick_color_value else "green"
            if any(value is not None for value in cvd.values()):
                serialized["cvd"] = cvd

            candle_color: Optional[str] = None
            open_value = serialized.get("open")
            close_value = serialized.get("close")
            if open_value is not None and close_value is not None:
                candle_color = "green" if close_value >= open_value else "red"
                serialized["candle_color"] = candle_color

            ema_20_value = serialized.get(self._ema_20_column)
            if ema_20_value is not None:
                serialized[f"_{self._ema_20_column}"] = ema_20_value

            serialized_rows.append(serialized)

        return serialized_rows


class VolumeFootprintRepository:
    """Builds queries for tradingview_volume_footprint rows."""

    def __init__(
        self,
        db: DatabaseClient,
        table: str = "tradingview_volume_footprint",
        symbol_column: str = "symbol",
        time_frame_column: str = "time_frame",
        timestamp_column: str = "timestamp",
        exchange_column: Optional[str] = None,
    ) -> None:
        self._db = db
        self._table = _sanitize_identifier(table, "tradingview_volume_footprint")
        self._symbol_column = _sanitize_identifier(symbol_column, "symbol")
        self._time_frame_column = _sanitize_identifier(time_frame_column, "time_frame")
        self._timestamp_column = _sanitize_identifier(timestamp_column, "timestamp")
        self._exchange_column = (
            _sanitize_identifier(exchange_column, "exchange") if exchange_column else None
        )

    @classmethod
    def from_env(cls, db: DatabaseClient) -> "VolumeFootprintRepository":
        """Build repository using optional env overrides."""

        return cls(
            db=db,
            table=os.getenv("FOOTPRINT_TABLE", "tradingview_volume_footprint"),
            symbol_column=os.getenv("FOOTPRINT_SYMBOL_COLUMN", "symbol"),
            time_frame_column=os.getenv("FOOTPRINT_TIME_FRAME_COLUMN", "time_frame"),
            timestamp_column=os.getenv("FOOTPRINT_TIMESTAMP_COLUMN", "timestamp"),
            exchange_column=os.getenv("FOOTPRINT_EXCHANGE_COLUMN"),
        )

    def fetch_volume_footprints(
        self,
        symbol: str,
        time_frame: str,
        limit: int,
        exchange: Optional[str] = None,
        start_timestamp: Optional[str] = None,
        end_timestamp: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return volume footprint rows with optional time filters."""

        limit = max(1, min(limit, 1000))
        time_frame = _normalize_time_frame(time_frame)

        select_columns = [
            "fp_id",
            f"{self._time_frame_column} AS time_frame",
            f"{self._symbol_column} AS symbol",
            f"{self._timestamp_column} AS timestamp",
        ]
        if self._exchange_column:
            select_columns.append(f"{self._exchange_column} AS exchange")
        select_columns.extend(
            [
                "poc",
                "vah",
                "val",
                "volume_delta",
                "levels",
                "total_fp_volume",
                "volume_diff",
                "created_at",
                "updated_at",
            ]
        )

        sql_parts = [
            f"SELECT {', '.join(select_columns)} FROM {self._table}",
            "WHERE {symbol_col} = %s AND {time_frame_col} = %s".format(
                symbol_col=self._symbol_column,
                time_frame_col=self._time_frame_column,
            ),
        ]
        params: List[Any] = [symbol, time_frame]

        if exchange is not None:
            if not self._exchange_column:
                raise ValueError("Exchange filtering requested but no exchange column configured.")
            sql_parts.append(f"AND {self._exchange_column} = %s")
            params.append(exchange)

        if start_timestamp is not None:
            sql_parts.append(f"AND {self._timestamp_column} >= %s")
            params.append(start_timestamp)
        if end_timestamp is not None:
            sql_parts.append(f"AND {self._timestamp_column} <= %s")
            params.append(end_timestamp)

        sql_parts.append(f"ORDER BY {self._timestamp_column} DESC")
        sql_parts.append("LIMIT %s")
        params.append(limit)

        rows = self._db.fetch_all(" ".join(sql_parts), params)
        rows.reverse()

        serialized_rows: List[Dict[str, Any]] = []
        for row in rows:
            serialized = _serialize_row(row)
            levels_value = serialized.get("levels")
            if isinstance(levels_value, (bytes, bytearray)):
                try:
                    levels_value = levels_value.decode("utf-8")
                    serialized["levels"] = levels_value
                except UnicodeDecodeError:
                    pass
            if isinstance(levels_value, str):
                try:
                    serialized["levels"] = json.loads(levels_value)
                except ValueError:
                    # Leave as-is if not JSON encoded.
                    pass
            serialized_rows.append(serialized)

        return serialized_rows


class CandleCvdRepository:
    """Builds queries for tradingview_candle_cvd rows."""

    def __init__(
        self,
        db: DatabaseClient,
        table: str = "tradingview_candle_cvd",
        symbol_column: str = "symbol",
        time_frame_column: str = "time_frame",
        timestamp_column: str = "timestamp",
        exchange_column: Optional[str] = "exchange",
    ) -> None:
        self._db = db
        self._table = _sanitize_identifier(table, "tradingview_candle_cvd")
        self._symbol_column = _sanitize_identifier(symbol_column, "symbol")
        self._time_frame_column = _sanitize_identifier(time_frame_column, "time_frame")
        self._timestamp_column = _sanitize_identifier(timestamp_column, "timestamp")
        self._exchange_column = (
            _sanitize_identifier(exchange_column, "exchange") if exchange_column else None
        )

    @classmethod
    def from_env(cls, db: DatabaseClient) -> "CandleCvdRepository":
        """Build repository using optional env overrides."""

        return cls(
            db=db,
            table=os.getenv("CVD_TABLE", "tradingview_candle_cvd"),
            symbol_column=os.getenv("CVD_SYMBOL_COLUMN", "symbol"),
            time_frame_column=os.getenv("CVD_TIME_FRAME_COLUMN", "time_frame"),
            timestamp_column=os.getenv("CVD_TIMESTAMP_COLUMN", "timestamp"),
            exchange_column=os.getenv("CVD_EXCHANGE_COLUMN", "exchange"),
        )

    def fetch_cvd(
        self,
        symbol: str,
        time_frame: str,
        limit: int,
        exchange: Optional[str] = None,
        start_timestamp: Optional[str] = None,
        end_timestamp: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return candle CVD rows with optional time filters."""

        limit = max(1, min(limit, 1000))
        time_frame = _normalize_time_frame(time_frame)

        select_columns = [
            "cvd_id",
            f"{self._exchange_column} AS exchange" if self._exchange_column else None,
            f"{self._symbol_column} AS symbol",
            f"{self._time_frame_column} AS time_frame",
            f"{self._timestamp_column} AS timestamp",
            "open",
            "high",
            "low",
            "close",
            "ohlc_color",
            "wick_color",
            "border_color",
        ]
        select_columns = [col for col in select_columns if col is not None]

        sql_parts = [
            f"SELECT {', '.join(select_columns)} FROM {self._table}",
            "WHERE {symbol_col} = %s AND {time_frame_col} = %s".format(
                symbol_col=self._symbol_column,
                time_frame_col=self._time_frame_column,
            ),
        ]
        params: List[Any] = [symbol, time_frame]

        if exchange is not None:
            if not self._exchange_column:
                raise ValueError("Exchange filtering requested but no exchange column configured.")
            sql_parts.append(f"AND {self._exchange_column} = %s")
            params.append(exchange)

        if start_timestamp is not None:
            sql_parts.append(f"AND {self._timestamp_column} >= %s")
            params.append(start_timestamp)
        if end_timestamp is not None:
            sql_parts.append(f"AND {self._timestamp_column} <= %s")
            params.append(end_timestamp)

        sql_parts.append(f"ORDER BY {self._timestamp_column} DESC")
        sql_parts.append("LIMIT %s")
        params.append(limit)

        rows = self._db.fetch_all(" ".join(sql_parts), params)
        rows.reverse()
        return [_serialize_row(row) for row in rows]


class EmaRepository:
    """Builds queries for tradingview_ema rows."""

    def __init__(
        self,
        db: DatabaseClient,
        table: str = "tradingview_ema",
        symbol_column: str = "symbol",
        time_frame_column: str = "time_frame",
        timestamp_column: str = "timestamp",
        exchange_column: Optional[str] = "exchange",
    ) -> None:
        self._db = db
        self._table = _sanitize_identifier(table, "tradingview_ema")
        self._symbol_column = _sanitize_identifier(symbol_column, "symbol")
        self._time_frame_column = _sanitize_identifier(time_frame_column, "time_frame")
        self._timestamp_column = _sanitize_identifier(timestamp_column, "timestamp")
        self._exchange_column = (
            _sanitize_identifier(exchange_column, "exchange") if exchange_column else None
        )

    @classmethod
    def from_env(cls, db: DatabaseClient) -> "EmaRepository":
        """Build repository using optional env overrides."""

        return cls(
            db=db,
            table=os.getenv("EMA_TABLE", "tradingview_ema"),
            symbol_column=os.getenv("EMA_SYMBOL_COLUMN", "symbol"),
            time_frame_column=os.getenv("EMA_TIME_FRAME_COLUMN", "time_frame"),
            timestamp_column=os.getenv("EMA_TIMESTAMP_COLUMN", "timestamp"),
            exchange_column=os.getenv("EMA_EXCHANGE_COLUMN", "exchange"),
        )

    def fetch_ema(
        self,
        symbol: str,
        time_frame: str,
        limit: int,
        exchange: Optional[str] = None,
        start_timestamp: Optional[str] = None,
        end_timestamp: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return EMA rows with optional time filters."""

        limit = max(1, min(limit, 1000))
        time_frame = _normalize_time_frame(time_frame)

        select_columns = [
            "e_id",
            f"{self._exchange_column} AS exchange" if self._exchange_column else None,
            f"{self._symbol_column} AS symbol",
            f"{self._time_frame_column} AS time_frame",
            f"{self._timestamp_column} AS timestamp",
            "20_ema",
            "date_time",
        ]
        select_columns = [col for col in select_columns if col is not None]

        sql_parts = [
            f"SELECT {', '.join(select_columns)} FROM {self._table}",
            "WHERE {symbol_col} = %s AND {time_frame_col} = %s".format(
                symbol_col=self._symbol_column,
                time_frame_col=self._time_frame_column,
            ),
        ]
        params: List[Any] = [symbol, time_frame]

        if exchange is not None:
            if not self._exchange_column:
                raise ValueError("Exchange filtering requested but no exchange column configured.")
            sql_parts.append(f"AND {self._exchange_column} = %s")
            params.append(exchange)

        if start_timestamp is not None:
            sql_parts.append(f"AND {self._timestamp_column} >= %s")
            params.append(start_timestamp)
        if end_timestamp is not None:
            sql_parts.append(f"AND {self._timestamp_column} <= %s")
            params.append(end_timestamp)

        sql_parts.append(f"ORDER BY {self._timestamp_column} DESC")
        sql_parts.append("LIMIT %s")
        params.append(limit)

        rows = self._db.fetch_all(" ".join(sql_parts), params)
        rows.reverse()
        return [_serialize_row(row) for row in rows]
