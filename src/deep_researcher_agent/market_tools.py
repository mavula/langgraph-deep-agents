"""Standalone market tools (trading + zones) copied from langgraph-server."""

from __future__ import annotations

import datetime as dt
import json
from array import array
from functools import lru_cache
from typing import Annotated, Any, Optional, TypedDict, Union

from langchain_core.tools import tool

from .config import DatabaseConfig
from .database import DatabaseClient
from .repository import (
    CandleCvdRepository,
    CandleRepository,
    EmaRepository,
    VolumeFootprintRepository,
)


# --- Trading tools ------------------------------------------------------- #

class CandleRow(TypedDict, total=False):
    data_id: int
    exchange: str
    symbol: str
    time_frame: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: str
    date_time: str
    volume_delta: float
    poc: float
    vah: float
    val: float
    footprint_levels: Union[dict[str, Any], list[Any], str, None]
    total_fp_volume: float
    volume_diff: float
    candle_color: str
    cvd: dict[str, Any]


class CandleResponse(TypedDict):
    start_timestamp: Optional[str]
    end_timestamp: Optional[str]
    count: int
    candles: list[CandleRow]


class EmaRow(TypedDict, total=False):
    e_id: int
    exchange: str
    symbol: str
    time_frame: str
    timestamp: str
    date_time: str
    _20_ema: float
    _50_ema: float
    _100_ema: float
    _200_ema: float


class EmaResponse(TypedDict):
    start_timestamp: Optional[str]
    end_timestamp: Optional[str]
    count: int
    ema: list[EmaRow]


class CurrentDateResponse(TypedDict):
    current_date: str


class CompareDatesResponse(TypedDict):
    relation: str


class ReportNoteRow(TypedDict, total=False):
    id: int
    symbol: str
    report_date: str
    notes: str
    embedding: Any
    source: Optional[str]
    tags: Any
    confidence_score: Optional[float]
    created_at: str
    updated_at: str


class ReportNotesResponse(TypedDict):
    count: int
    notes: list[ReportNoteRow]


@lru_cache
def _get_candle_repository() -> CandleRepository:
    """Return a cached candle repository backed by env configuration."""
    config = DatabaseConfig.from_env()
    db_client = DatabaseClient(config)
    return CandleRepository.from_env(db_client)


@lru_cache
def _get_ema_repository() -> EmaRepository:
    """Return a cached EMA repository backed by env configuration."""
    config = DatabaseConfig.from_env()
    db_client = DatabaseClient(config)
    return EmaRepository.from_env(db_client)


def _format_timestamp(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    try:
        dt_value = dt.datetime.fromisoformat(value)
    except ValueError:
        return value
    return dt_value.strftime("%Y-%m-%d %H:%M:%S")


def _format_candle_response(rows: list[CandleRow]) -> CandleResponse:
    start_iso = rows[0]["timestamp"] if rows else None
    end_iso = rows[-1]["timestamp"] if rows else None
    return {
        "start_timestamp": _format_timestamp(start_iso),
        "end_timestamp": _format_timestamp(end_iso),
        "count": len(rows),
        "candles": rows,
    }


def _format_ema_response(rows: list[EmaRow]) -> EmaResponse:
    start_iso = rows[0]["timestamp"] if rows else None
    end_iso = rows[-1]["timestamp"] if rows else None
    normalized_rows: list[EmaRow] = []
    for row in rows:
        normalized_rows.append(
            {
                **row,
                "_20_ema": row.get("20_ema"),  # type: ignore[literal-required]
                "_50_ema": row.get("50_ema"),  # type: ignore[literal-required]
                "_100_ema": row.get("100_ema"),  # type: ignore[literal-required]
                "_200_ema": row.get("200_ema"),  # type: ignore[literal-required]
            }
        )
    return {
        "start_timestamp": _format_timestamp(start_iso),
        "end_timestamp": _format_timestamp(end_iso),
        "count": len(rows),
        "ema": normalized_rows,
    }


@tool
def get_candles(
    symbol: Annotated[str, "Ticker or instrument identifier exactly as stored in the DB."],
    time_frame: Annotated[
        str,
        "Time frame value stored in the time_frame column (e.g. 1, 5, 60, 1D). '30' is normalized to '30m'.",
    ],
    limit: Annotated[int, "Maximum number of rows to return (1-1000)."] = 200,
    exchange: Annotated[Optional[str], "Optional exchange to filter if multiple venues store the same symbol."] = None,
    start_timestamp: Annotated[
        Optional[str],
        "Inclusive timestamp filter in 'YYYY-MM-DD HH:MM:SS' format.",
    ] = None,
    end_timestamp: Annotated[
        Optional[str],
        "Inclusive timestamp upper bound in 'YYYY-MM-DD HH:MM:SS' format.",
    ] = None,
) -> CandleResponse:
    """Fetch OHLCV candles for the given symbol/resolution."""

    repository = _get_candle_repository()
    rows = repository.fetch_candles(
        symbol=symbol,
        time_frame=time_frame,
        limit=limit,
        exchange=exchange,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
    )
    return _format_candle_response(rows)


@tool
def get_ema(
    symbol: Annotated[str, "Ticker or instrument identifier exactly as stored in the DB."],
    time_frame: Annotated[
        str,
        "Time frame value stored in the time_frame column (e.g. 1, 5, 60, 1D). '30' is normalized to '30m'.",
    ],
    limit: Annotated[int, "Maximum number of rows to return (1-1000)."] = 200,
    exchange: Annotated[Optional[str], "Optional exchange to filter if multiple venues store the same symbol."] = None,
    start_timestamp: Annotated[
        Optional[str],
        "Inclusive timestamp filter in 'YYYY-MM-DD HH:MM:SS' format.",
    ] = None,
    end_timestamp: Annotated[
        Optional[str],
        "Inclusive timestamp upper bound in 'YYYY-MM-DD HH:MM:SS' format.",
    ] = None,
) -> EmaResponse:
    """Fetch EMA values for the given symbol/resolution."""

    repository = _get_ema_repository()
    rows = repository.fetch_ema(
        symbol=symbol,
        time_frame=time_frame,
        limit=limit,
        exchange=exchange,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
    )
    return _format_ema_response(rows)


@tool
def get_current_date() -> CurrentDateResponse:
    """Return today's date in ISO format."""
    today = dt.date.today()
    return {"current_date": today.isoformat()}


@tool
def compare_dates(
    requested_date: Annotated[str, "Requested date in ISO format YYYY-MM-DD."],
    current_date: Annotated[str, "Current date in ISO format YYYY-MM-DD."],
) -> CompareDatesResponse:
    """Compare the requested date to the provided current date."""
    try:
        requested = dt.date.fromisoformat(requested_date)
        current = dt.date.fromisoformat(current_date)
    except ValueError as exc:
        raise ValueError("Dates must be in ISO format YYYY-MM-DD.") from exc
    if requested == current:
        relation = "today"
    elif requested < current:
        relation = "past"
    else:
        relation = "future"
    return {"relation": relation}


# --- Zone tools ---------------------------------------------------------- #

class ZoneRepository:
    """Simple repository for inserting and updating ai_zones rows."""

    def __init__(self, db: DatabaseClient, table: str = "ai_zones") -> None:
        self._db = db
        self._table = table

    @classmethod
    def from_env(cls, db: DatabaseClient) -> "ZoneRepository":
        return cls(db=db, table="ai_zones")

    def insert_zone(self, payload: dict[str, Any]) -> int:
        columns = []
        placeholders = []
        values: list[Any] = []
        for key, value in payload.items():
            if value is None:
                continue
            columns.append(key)
            placeholders.append("%s")
            values.append(value)

        sql = f"INSERT INTO {self._table} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
        with self._db.connection() as conn, conn.cursor() as cursor:
            cursor.execute(sql, tuple(values))
            return int(cursor.lastrowid)

    def update_zone(self, zone_id: int, updates: dict[str, Any]) -> int:
        sets = []
        values: list[Any] = []
        for key, value in updates.items():
            if value is None:
                continue
            sets.append(f"{key} = %s")
            values.append(value)

        if not sets:
            raise ValueError("No updates provided.")

        values.append(zone_id)
        sql = f"UPDATE {self._table} SET {', '.join(sets)} WHERE id = %s"
        with self._db.connection() as conn, conn.cursor() as cursor:
            cursor.execute(sql, tuple(values))
            return cursor.rowcount

    def fetch_zones(
        self,
        zone_id: Optional[int] = None,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        status: Optional[str] = None,
        zone_type: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        select_columns = [
            "z.*",
            "t.id AS touch_id",
            "t.touch_number AS touch_number",
            "t.touch_start_ts AS touch_start_ts",
            "t.touch_end_ts AS touch_end_ts",
            "t.touch_price_low AS touch_price_low",
            "t.touch_price_high AS touch_price_high",
            "t.touch_price_close AS touch_price_close",
            "t.touch_type AS touch_type",
            "t.zone_status_before AS touch_zone_status_before",
            "t.zone_status_after AS touch_zone_status_after",
            "t.reaction_outcome AS touch_reaction_outcome",
            "t.reaction_comment AS touch_reaction_comment",
        ]

        sql_parts = [
            f"SELECT {', '.join(select_columns)}",
            f"FROM {self._table} AS z",
            "LEFT JOIN ai_zone_touches AS t ON t.zone_id = z.id",
            "WHERE 1=1",
        ]
        params: list[Any] = []

        if zone_id:
            sql_parts.append("AND z.id = %s")
            params.append(zone_id)
        if symbol:
            sql_parts.append("AND z.symbol = %s")
            params.append(symbol)
        if timeframe:
            sql_parts.append("AND z.timeframe = %s")
            params.append(timeframe)
        if status:
            sql_parts.append("AND z.status = %s")
            params.append(status)
        if zone_type:
            sql_parts.append("AND z.zone_type = %s")
            params.append(zone_type)

        sql_parts.append("ORDER BY anchor_start_ts DESC")
        sql_parts.append("LIMIT %s")
        params.append(max(1, min(limit, 500)))

        with self._db.connection() as conn, conn.cursor() as cursor:
            cursor.execute(" ".join(sql_parts), tuple(params))
            return list(cursor.fetchall())


class ZoneNoteRepository:
    """Repository for inserting zone notes into ai_zone_notes."""

    def __init__(self, db: DatabaseClient, table: str = "ai_zone_notes") -> None:
        self._db = db
        self._table = table

    @classmethod
    def from_env(cls, db: DatabaseClient) -> "ZoneNoteRepository":
        return cls(db=db, table="ai_zone_notes")

    def insert_note(self, payload: dict[str, Any]) -> int:
        columns = []
        placeholders = []
        values: list[Any] = []
        for key, value in payload.items():
            if value is None:
                continue
            columns.append(key)
            placeholders.append("%s")
            values.append(value)

        if not columns:
            raise ValueError("No values provided for ai_zone_notes insert.")

        sql = f"INSERT INTO {self._table} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
        with self._db.connection() as conn, conn.cursor() as cursor:
            cursor.execute(sql, tuple(values))
            return int(cursor.lastrowid)


class ZoneTouchRepository:
    """Repository for inserting zone touches into ai_zone_touches."""

    def __init__(self, db: DatabaseClient, table: str = "ai_zone_touches") -> None:
        self._db = db
        self._table = table

    @classmethod
    def from_env(cls, db: DatabaseClient) -> "ZoneTouchRepository":
        return cls(db=db, table="ai_zone_touches")

    def insert_touch(self, payload: dict[str, Any]) -> int:
        columns = []
        placeholders = []
        values: list[Any] = []
        for key, value in payload.items():
            if value is None:
                continue
            columns.append(key)
            placeholders.append("%s")
            values.append(value)

        if not columns:
            raise ValueError("No values provided for ai_zone_touches insert.")

        sql = f"INSERT INTO {self._table} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
        with self._db.connection() as conn, conn.cursor() as cursor:
            cursor.execute(sql, tuple(values))
            return int(cursor.lastrowid)


class ZoneTouchPriceActionRepository:
    """Repository for inserting touch price actions into ai_zone_touch_price_actions."""

    def __init__(self, db: DatabaseClient, table: str = "ai_zone_touch_price_actions") -> None:
        self._db = db
        self._table = table

    @classmethod
    def from_env(cls, db: DatabaseClient) -> "ZoneTouchPriceActionRepository":
        return cls(db=db, table="ai_zone_touch_price_actions")

    def insert_action(self, payload: dict[str, Any]) -> int:
        columns = []
        placeholders = []
        values: list[Any] = []
        for key, value in payload.items():
            if value is None:
                continue
            columns.append(key)
            placeholders.append("%s")
            values.append(value)

        if not columns:
            raise ValueError("No values provided for ai_zone_touch_price_actions insert.")

        sql = f"INSERT INTO {self._table} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
        with self._db.connection() as conn, conn.cursor() as cursor:
            cursor.execute(sql, tuple(values))
            return int(cursor.lastrowid)

    def update_action(self, action_id: int, updates: dict[str, Any]) -> int:
        sets = []
        values: list[Any] = []
        for key, value in updates.items():
            if value is None:
                continue
            sets.append(f"{key} = %s")
            values.append(value)

        if not sets:
            raise ValueError("No updates provided for ai_zone_touch_price_actions.")

        values.append(action_id)
        sql = f"UPDATE {self._table} SET {', '.join(sets)} WHERE id = %s"
        with self._db.connection() as conn, conn.cursor() as cursor:
            cursor.execute(sql, tuple(values))
            return cursor.rowcount


class DoubleTopPatternRepository:
    """Repository for inserting double top patterns into ai_double_top_patterns."""

    def __init__(self, db: DatabaseClient, table: str = "ai_double_top_patterns") -> None:
        self._db = db
        self._table = table

    @classmethod
    def from_env(cls, db: DatabaseClient) -> "DoubleTopPatternRepository":
        return cls(db=db, table="ai_double_top_patterns")

    def insert_pattern(self, payload: dict[str, Any]) -> int:
        columns = []
        placeholders = []
        values: list[Any] = []
        for key, value in payload.items():
            if value is None:
                continue
            columns.append(key)
            placeholders.append("%s")
            values.append(value)

        if not columns:
            raise ValueError("No values provided for ai_double_top_patterns insert.")

        sql = f"INSERT INTO {self._table} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
        with self._db.connection() as conn, conn.cursor() as cursor:
            cursor.execute(sql, tuple(values))
            return int(cursor.lastrowid)

    def update_pattern(self, pattern_id: int, updates: dict[str, Any]) -> int:
        sets = []
        values: list[Any] = []
        for key, value in updates.items():
            if value is None:
                continue
            sets.append(f"{key} = %s")
            values.append(value)

        if not sets:
            raise ValueError("No updates provided for ai_ema_20_patterns.")

        values.append(pattern_id)
        sql = f"UPDATE {self._table} SET {', '.join(sets)} WHERE id = %s"
        with self._db.connection() as conn, conn.cursor() as cursor:
            cursor.execute(sql, tuple(values))
            return cursor.rowcount

    def update_pattern(self, pattern_id: int, updates: dict[str, Any]) -> int:
        sets = []
        values: list[Any] = []
        for key, value in updates.items():
            if value is None:
                continue
            sets.append(f"{key} = %s")
            values.append(value)

        if not sets:
            raise ValueError("No updates provided for ai_double_top_patterns.")

        values.append(pattern_id)
        sql = f"UPDATE {self._table} SET {', '.join(sets)} WHERE id = %s"
        with self._db.connection() as conn, conn.cursor() as cursor:
            cursor.execute(sql, tuple(values))
            return cursor.rowcount


class DoubleBottomPatternRepository:
    """Repository for inserting double bottom patterns into ai_double_bottom_patterns."""

    def __init__(self, db: DatabaseClient, table: str = "ai_double_bottom_patterns") -> None:
        self._db = db
        self._table = table

    @classmethod
    def from_env(cls, db: DatabaseClient) -> "DoubleBottomPatternRepository":
        return cls(db=db, table="ai_double_bottom_patterns")

    def insert_pattern(self, payload: dict[str, Any]) -> int:
        columns = []
        placeholders = []
        values: list[Any] = []
        for key, value in payload.items():
            if value is None:
                continue
            columns.append(key)
            placeholders.append("%s")
            values.append(value)

        if not columns:
            raise ValueError("No values provided for ai_double_bottom_patterns insert.")

        sql = f"INSERT INTO {self._table} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
        with self._db.connection() as conn, conn.cursor() as cursor:
            cursor.execute(sql, tuple(values))
            return int(cursor.lastrowid)

    def update_pattern(self, pattern_id: int, updates: dict[str, Any]) -> int:
        sets = []
        values: list[Any] = []
        for key, value in updates.items():
            if value is None:
                continue
            sets.append(f"{key} = %s")
            values.append(value)

        if not sets:
            raise ValueError("No updates provided for ai_double_bottom_patterns.")

        values.append(pattern_id)
        sql = f"UPDATE {self._table} SET {', '.join(sets)} WHERE id = %s"
        with self._db.connection() as conn, conn.cursor() as cursor:
            cursor.execute(sql, tuple(values))
            return cursor.rowcount


class VTopPatternRepository:
    """Repository for inserting V top patterns into ai_v_top_patterns."""

    def __init__(self, db: DatabaseClient, table: str = "ai_v_top_patterns") -> None:
        self._db = db
        self._table = table

    @classmethod
    def from_env(cls, db: DatabaseClient) -> "VTopPatternRepository":
        return cls(db=db, table="ai_v_top_patterns")

    def insert_pattern(self, payload: dict[str, Any]) -> int:
        columns = []
        placeholders = []
        values: list[Any] = []
        for key, value in payload.items():
            if value is None:
                continue
            columns.append(key)
            placeholders.append("%s")
            values.append(value)

        if not columns:
            raise ValueError("No values provided for ai_v_top_patterns insert.")

        sql = f"INSERT INTO {self._table} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
        with self._db.connection() as conn, conn.cursor() as cursor:
            cursor.execute(sql, tuple(values))
            return int(cursor.lastrowid)

    def update_pattern(self, pattern_id: int, updates: dict[str, Any]) -> int:
        sets = []
        values: list[Any] = []
        for key, value in updates.items():
            if value is None:
                continue
            sets.append(f"{key} = %s")
            values.append(value)

        if not sets:
            raise ValueError("No updates provided for ai_v_top_patterns.")

        values.append(pattern_id)
        sql = f"UPDATE {self._table} SET {', '.join(sets)} WHERE id = %s"
        with self._db.connection() as conn, conn.cursor() as cursor:
            cursor.execute(sql, tuple(values))
            return cursor.rowcount


class Ema20PatternRepository:
    """Repository for inserting EMA-20 patterns into ai_ema_20_patterns."""

    def __init__(self, db: DatabaseClient, table: str = "ai_ema_20_patterns") -> None:
        self._db = db
        self._table = table

    @classmethod
    def from_env(cls, db: DatabaseClient) -> "Ema20PatternRepository":
        return cls(db=db, table="ai_ema_20_patterns")

    def insert_pattern(self, payload: dict[str, Any]) -> int:
        columns = []
        placeholders = []
        values: list[Any] = []
        for key, value in payload.items():
            if value is None:
                continue
            columns.append(key)
            placeholders.append("%s")
            values.append(value)

        if not columns:
            raise ValueError("No values provided for ai_ema_20_patterns insert.")

        sql = f"INSERT INTO {self._table} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
        with self._db.connection() as conn, conn.cursor() as cursor:
            cursor.execute(sql, tuple(values))
            return int(cursor.lastrowid)


class ZoneRelationshipRepository:
    """Repository for inserting zone relationships into ai_zone_relationships."""

    def __init__(self, db: DatabaseClient, table: str = "ai_zone_relationships") -> None:
        self._db = db
        self._table = table

    @classmethod
    def from_env(cls, db: DatabaseClient) -> "ZoneRelationshipRepository":
        return cls(db=db, table="ai_zone_relationships")

    def insert_relationship(self, payload: dict[str, Any]) -> int:
        columns = []
        placeholders = []
        values: list[Any] = []
        for key, value in payload.items():
            if value is None:
                continue
            columns.append(key)
            placeholders.append("%s")
            values.append(value)

        if not columns:
            raise ValueError("No values provided for ai_zone_relationships insert.")

        sql = f"INSERT INTO {self._table} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
        with self._db.connection() as conn, conn.cursor() as cursor:
            cursor.execute(sql, tuple(values))
            return int(cursor.lastrowid)


class ReportNoteRepository:
    """Repository for fetching and updating ai_report_notes rows."""

    def __init__(self, db: DatabaseClient, table: str = "ai_report_notes") -> None:
        self._db = db
        self._table = table

    @classmethod
    def from_env(cls, db: DatabaseClient) -> "ReportNoteRepository":
        return cls(db=db, table="ai_report_notes")

    def fetch_notes(
        self,
        note_id: Optional[int] = None,
        symbol: Optional[str] = None,
        report_date: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        sql_parts = [
            "SELECT id, symbol, report_date, notes, embedding, source, tags, confidence_score, created_at, updated_at",
            f"FROM {self._table}",
            "WHERE 1=1",
        ]
        params: list[Any] = []

        if note_id:
            sql_parts.append("AND id = %s")
            params.append(note_id)
        if symbol:
            sql_parts.append("AND symbol = %s")
            params.append(symbol)
        if report_date:
            sql_parts.append("AND report_date = %s")
            params.append(report_date)

        sql_parts.append("ORDER BY updated_at DESC")
        sql_parts.append("LIMIT %s")
        params.append(max(1, min(limit, 500)))

        with self._db.connection() as conn, conn.cursor() as cursor:
            cursor.execute(" ".join(sql_parts), tuple(params))
            return list(cursor.fetchall())

    def insert_note(self, payload: dict[str, Any]) -> int:
        columns = []
        placeholders = []
        values: list[Any] = []
        for key, value in payload.items():
            if value is None:
                continue
            columns.append(key)
            placeholders.append("%s")
            values.append(value)

        if not columns:
            raise ValueError("No values provided for ai_report_notes insert.")

        sql = f"INSERT INTO {self._table} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
        with self._db.connection() as conn, conn.cursor() as cursor:
            cursor.execute(sql, tuple(values))
            return int(cursor.lastrowid)

    def update_note(self, note_id: int, updates: dict[str, Any]) -> int:
        sets = []
        values: list[Any] = []
        for key, value in updates.items():
            if value is None:
                continue
            sets.append(f"{key} = %s")
            values.append(value)

        if not sets:
            raise ValueError("No updates provided for ai_report_notes.")

        values.append(note_id)
        sql = f"UPDATE {self._table} SET {', '.join(sets)} WHERE id = %s"
        with self._db.connection() as conn, conn.cursor() as cursor:
            cursor.execute(sql, tuple(values))
            return cursor.rowcount


def _get_report_note_repo() -> ReportNoteRepository:
    config = DatabaseConfig.from_env()
    db_client = DatabaseClient(config)
    return ReportNoteRepository.from_env(db_client)


def _get_zone_repo() -> ZoneRepository:
    config = DatabaseConfig.from_env()
    db_client = DatabaseClient(config)
    return ZoneRepository.from_env(db_client)


def _get_zone_note_repo() -> ZoneNoteRepository:
    config = DatabaseConfig.from_env()
    db_client = DatabaseClient(config)
    return ZoneNoteRepository.from_env(db_client)


def _get_zone_touch_repo() -> ZoneTouchRepository:
    config = DatabaseConfig.from_env()
    db_client = DatabaseClient(config)
    return ZoneTouchRepository.from_env(db_client)


def _get_zone_touch_price_action_repo() -> ZoneTouchPriceActionRepository:
    config = DatabaseConfig.from_env()
    db_client = DatabaseClient(config)
    return ZoneTouchPriceActionRepository.from_env(db_client)


def _get_double_top_pattern_repo() -> DoubleTopPatternRepository:
    config = DatabaseConfig.from_env()
    db_client = DatabaseClient(config)
    return DoubleTopPatternRepository.from_env(db_client)


def _get_double_bottom_pattern_repo() -> DoubleBottomPatternRepository:
    config = DatabaseConfig.from_env()
    db_client = DatabaseClient(config)
    return DoubleBottomPatternRepository.from_env(db_client)


def _get_v_top_pattern_repo() -> VTopPatternRepository:
    config = DatabaseConfig.from_env()
    db_client = DatabaseClient(config)
    return VTopPatternRepository.from_env(db_client)


def _get_ema20_pattern_repo() -> Ema20PatternRepository:
    config = DatabaseConfig.from_env()
    db_client = DatabaseClient(config)
    return Ema20PatternRepository.from_env(db_client)


def _get_zone_relationship_repo() -> ZoneRelationshipRepository:
    config = DatabaseConfig.from_env()
    db_client = DatabaseClient(config)
    return ZoneRelationshipRepository.from_env(db_client)


def _coerce_bool(value: Optional[bool | int | str]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return 1 if value != 0 else 0
    if isinstance(value, str):
        return 1 if value.strip().lower() in {"1", "true", "yes", "y"} else 0
    return None


@tool
def create_zone(
    symbol: Annotated[str, "Symbol identifier (e.g., BANKNIFTY1!)."],
    timeframe: Annotated[str, "Timeframe enum value: 1m,3m,5m,10m,15m,30m,1H,4H,1D."],
    zone_type: Annotated[str, "Zone type enum: DEMAND or SUPPLY."],
    zone_role: Annotated[str, "Zone role enum: ORIGIN, RETEST, or FLIP."],
    price_low: Annotated[float, "Low price of the zone."],
    price_high: Annotated[float, "High price of the zone."],
    anchor_start_ts: Annotated[str, "Anchor start timestamp (YYYY-MM-DD HH:MM:SS)."],
    anchor_end_ts: Annotated[str, "Anchor end timestamp (YYYY-MM-DD HH:MM:SS)."],
    status: Annotated[str, "Status enum: PENDING, ACTIVE, TOUCHED, VIOLATED, INVALIDATED."] = "PENDING",
    structure_pattern: Annotated[
        str,
        "Pattern enum: RALLY_BASE_RALLY, DROP_BASE_RALLY, RALLY_BASE_DROP, DROP_BASE_DROP, OTHER.",
    ] = "OTHER",
    failed_low_attempts: Annotated[int, "Number of failed low attempts."] = 0,
    displacement_strength: Annotated[int, "Displacement strength metric."] = 0,
    has_bullish_delta_divergence: Annotated[Optional[bool], "1 if bullish delta divergence present."] = None,
    delta_low_price: Annotated[Optional[float], "Delta low price."] = None,
    delta_low_value: Annotated[Optional[int], "Delta low value."] = None,
    delta_prev_low_value: Annotated[Optional[int], "Delta previous low value."] = None,
    delta_comment: Annotated[Optional[str], "Comment about delta."] = None,
    poc_tf: Annotated[Optional[str], "POC timeframe enum: SESSION, DAY, CUSTOM."] = "DAY",
    poc_initial_price: Annotated[Optional[float], "Initial POC price."] = None,
    poc_migrated_price: Annotated[Optional[float], "Migrated POC price."] = None,
    poc_migration_confirmed: Annotated[Optional[bool], "Whether POC migration confirmed."] = None,
    cvd_confirmation_type: Annotated[
        Optional[str],
        "CVD confirmation enum: NONE, RISING_AWAY, DIVERGENCE_RETEST, ABSORPTION.",
    ] = "NONE",
    cvd_absorption_side: Annotated[
        Optional[str],
        "CVD absorption side enum: NONE, BUY, SELL.",
    ] = "NONE",
    cvd_value_at_origin: Annotated[Optional[int], "CVD value at origin."] = None,
    cvd_value_at_exit: Annotated[Optional[int], "CVD value at exit."] = None,
    cvd_comment: Annotated[Optional[str], "CVD comment."] = None,
    confidence_score: Annotated[int, "Confidence score 0-255."] = 0,
    notes: Annotated[Optional[str], "Freeform notes."] = None,
) -> dict:
    """Insert a new demand/supply zone into ai_zones."""

    repo = _get_zone_repo()
    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "zone_type": zone_type,
        "zone_role": zone_role,
        "status": status,
        "price_low": price_low,
        "price_high": price_high,
        "anchor_start_ts": anchor_start_ts,
        "anchor_end_ts": anchor_end_ts,
        "structure_pattern": structure_pattern,
        "failed_low_attempts": failed_low_attempts,
        "displacement_strength": displacement_strength,
        "has_bullish_delta_divergence": _coerce_bool(has_bullish_delta_divergence),
        "delta_low_price": delta_low_price,
        "delta_low_value": delta_low_value,
        "delta_prev_low_value": delta_prev_low_value,
        "delta_comment": delta_comment,
        "poc_tf": poc_tf,
        "poc_initial_price": poc_initial_price,
        "poc_migrated_price": poc_migrated_price,
        "poc_migration_confirmed": _coerce_bool(poc_migration_confirmed),
        "cvd_confirmation_type": cvd_confirmation_type,
        "cvd_absorption_side": cvd_absorption_side,
        "cvd_value_at_origin": cvd_value_at_origin,
        "cvd_value_at_exit": cvd_value_at_exit,
        "cvd_comment": cvd_comment,
        "confidence_score": confidence_score,
        "notes": notes,
    }
    zone_id = repo.insert_zone(payload)
    return {"zone_id": zone_id, "status": "created"}


@tool
def update_zone(
    zone_id: Annotated[int, "Existing ai_zones.id to update."],
    status: Annotated[Optional[str], "Optional status update."] = None,
    confidence_score: Annotated[Optional[int], "Optional confidence score update."] = None,
    notes: Annotated[Optional[str], "Optional notes update."] = None,
    poc_migrated_price: Annotated[Optional[float], "Optional migrated POC price update."] = None,
    poc_migration_confirmed: Annotated[Optional[bool], "Optional POC migration confirmation flag."] = None,
    cvd_confirmation_type: Annotated[
        Optional[str],
        "Optional CVD confirmation enum: NONE, RISING_AWAY, DIVERGENCE_RETEST, ABSORPTION.",
    ] = None,
    cvd_absorption_side: Annotated[Optional[str], "Optional CVD absorption side: NONE, BUY, SELL."] = None,
    cvd_comment: Annotated[Optional[str], "Optional CVD comment update."] = None,
) -> dict:
    """Update selected fields on an existing ai_zones row."""

    repo = _get_zone_repo()
    updates = {
        "status": status,
        "confidence_score": confidence_score,
        "notes": notes,
        "poc_migrated_price": poc_migrated_price,
        "poc_migration_confirmed": _coerce_bool(poc_migration_confirmed),
        "cvd_confirmation_type": cvd_confirmation_type,
        "cvd_absorption_side": cvd_absorption_side,
        "cvd_comment": cvd_comment,
    }
    rows = repo.update_zone(zone_id, updates)
    return {"zone_id": zone_id, "rows_updated": rows}


@tool
def get_zones(
    zone_id: Annotated[Optional[int], "Optional zone id filter."] = None,
    symbol: Annotated[Optional[str], "Optional symbol filter."] = None,
    timeframe: Annotated[Optional[str], "Optional timeframe filter."] = None,
    status: Annotated[Optional[str], "Optional status filter."] = None,
    zone_type: Annotated[Optional[str], "Optional zone type filter: DEMAND or SUPPLY."] = None,
    limit: Annotated[int, "Maximum rows to return (1-500)."] = 50,
) -> list[dict]:
    """Fetch zones from ai_zones with optional filters."""

    repo = _get_zone_repo()
    return repo.fetch_zones(
        zone_id=zone_id,
        symbol=symbol,
        timeframe=timeframe,
        status=status,
        zone_type=zone_type,
        limit=limit,
    )


def _normalize_tags(tags: Optional[str | list[str]]) -> Optional[str]:
    if tags is None:
        return None
    if isinstance(tags, list):
        return ",".join(tag.strip() for tag in tags if tag.strip())
    return tags


def _serialize_embedding(embedding: Optional[bytes | bytearray | list[float] | tuple[float, ...] | str]) -> Optional[bytes]:
    if embedding is None:
        return None
    if isinstance(embedding, (bytes, bytearray)):
        return bytes(embedding)[:8192]
    if isinstance(embedding, str):
        return embedding.encode("utf-8")[:8192]

    floats = array("f", [float(value) for value in embedding])
    return floats.tobytes()[:8192]


def _serialize_metadata(metadata: Optional[dict[str, Any] | list[Any] | str]) -> Optional[str]:
    if metadata is None:
        return None
    if isinstance(metadata, str):
        return metadata
    try:
        return json.dumps(metadata)
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must be JSON-serializable") from exc


def _serialize_json_column(value: Optional[Any]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            json.loads(stripped)
        except ValueError:
            return json.dumps(stripped)
        else:
            return stripped
    try:
        return json.dumps(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Value must be valid JSON or JSON-serializable.") from exc


def _parse_json_column(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return json.loads(stripped)
        except ValueError:
            return value
    return value


def _format_date_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date().isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    return str(value)


def _format_datetime_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, dt.date):
        return value.isoformat()
    return str(value)


def _format_report_note_row(row: dict[str, Any]) -> ReportNoteRow:
    return {
        "id": row.get("id"),
        "symbol": row.get("symbol"),
        "report_date": _format_date_value(row.get("report_date")),
        "notes": row.get("notes"),
        "embedding": _parse_json_column(row.get("embedding")),
        "source": row.get("source"),
        "tags": _parse_json_column(row.get("tags")),
        "confidence_score": row.get("confidence_score"),
        "created_at": _format_datetime_value(row.get("created_at")),
        "updated_at": _format_datetime_value(row.get("updated_at")),
    }


@tool
def add_report_note(
    symbol: Annotated[str, "Symbol/ticker for the report note (e.g., BANKNIFTY1!)."],
    report_date: Annotated[str, "Report date in YYYY-MM-DD."],
    notes: Annotated[str, "Primary narrative/notes content to store."],
    embedding: Annotated[
        Optional[Any],
        "Optional embedding JSON (list/array) or JSON string.",
    ] = None,
    source: Annotated[Optional[str], "Optional source label (e.g., agent name or feed)."] = None,
    tags: Annotated[
        Optional[Any],
        "Optional tags JSON (list or object) or JSON string.",
    ] = None,
    confidence_score: Annotated[Optional[float], "Optional confidence score (0.00-9.99)."] = None,
) -> dict[str, Any]:
    """Insert a report note row into ai_report_notes."""

    repo = _get_report_note_repo()
    payload = {
        "symbol": symbol,
        "report_date": report_date,
        "notes": notes,
        "embedding": _serialize_json_column(embedding),
        "source": source,
        "tags": _serialize_json_column(tags),
        "confidence_score": confidence_score,
    }
    note_id = repo.insert_note(payload)
    return {"note_id": note_id, "status": "created"}


@tool
def get_report_notes(
    note_id: Annotated[Optional[int], "Optional ai_report_notes.id filter."] = None,
    symbol: Annotated[Optional[str], "Optional symbol filter."] = None,
    report_date: Annotated[Optional[str], "Optional report date filter in YYYY-MM-DD."] = None,
    limit: Annotated[int, "Maximum rows to return (1-500)."] = 50,
) -> ReportNotesResponse:
    """Fetch research report notes with optional filters."""

    repo = _get_report_note_repo()
    rows = repo.fetch_notes(
        note_id=note_id,
        symbol=symbol,
        report_date=report_date,
        limit=limit,
    )
    notes = [_format_report_note_row(row) for row in rows]
    return {"count": len(notes), "notes": notes}


@tool
def update_report_note(
    note_id: Annotated[int, "Existing ai_report_notes.id to update."],
    notes: Annotated[Optional[str], "Updated notes content."] = None,
    embedding: Annotated[
        Optional[Any],
        "Updated embedding JSON (list/array) or JSON string.",
    ] = None,
    source: Annotated[Optional[str], "Updated source label (e.g., feed name or agent)."] = None,
    tags: Annotated[
        Optional[Any],
        "Updated tags JSON (list or object) or JSON string.",
    ] = None,
    confidence_score: Annotated[Optional[float], "Optional confidence score (0.00-9.99)."] = None,
) -> dict[str, Any]:
    """Update selected fields on an existing ai_report_notes row."""

    repo = _get_report_note_repo()
    updates = {
        "notes": notes,
        "embedding": _serialize_json_column(embedding),
        "source": source,
        "tags": _serialize_json_column(tags),
        "confidence_score": confidence_score,
    }
    rows = repo.update_note(note_id, updates)
    return {"note_id": note_id, "rows_updated": rows}


@tool
def add_zone_note(
    zone_id: Annotated[int, "Existing ai_zones.id to attach the note to."],
    symbol: Annotated[str, "Symbol identifier that matches the zone."],
    timeframe: Annotated[str, "Timeframe string that matches the zone (e.g., 1H, 4H, 1D)."],
    content: Annotated[str, "Detailed note content to persist for the zone."],
    title: Annotated[Optional[str], "Optional note title."] = None,
    tags: Annotated[Optional[str | list[str]], "Optional tags as comma-separated string or list."] = None,
    author: Annotated[Optional[str], "Optional author identifier."] = None,
    source: Annotated[Optional[str], "Optional source label (e.g., agent, manual)."] = None,
    embedding: Annotated[
        Optional[bytes | bytearray | list[float] | tuple[float, ...] | str],
        "Optional serialized embedding for RAG. Bytes, utf-8 string, or list/tuple of floats (float32).",
    ] = None,
    metadata: Annotated[
        Optional[dict[str, Any] | list[Any] | str],
        "Optional JSON metadata for the note (stored in metadata_json).",
    ] = None,
) -> dict[str, Any]:
    """Insert a detailed note for a specific zone into ai_zone_notes."""

    repo = _get_zone_note_repo()
    payload = {
        "zone_id": zone_id,
        "symbol": symbol,
        "timeframe": timeframe,
        "title": title,
        "content": content,
        "tags": _normalize_tags(tags),
        "author": author,
        "source": source,
        "embedding": _serialize_embedding(embedding),
        "metadata_json": _serialize_metadata(metadata),
    }
    note_id = repo.insert_note(payload)
    return {"note_id": note_id, "status": "created"}


@tool
def add_zone_touch(
    zone_id: Annotated[int, "Existing ai_zones.id to attach the touch to."],
    symbol: Annotated[str, "Symbol identifier that matches the zone."],
    timeframe: Annotated[str, "Timeframe enum value: 1m,3m,5m,10m,15m,30m,1H,4H,1D."],
    touch_number: Annotated[int, "Ordinal touch number for the zone (1-based)."],
    touch_start_ts: Annotated[str, "Touch start timestamp (YYYY-MM-DD HH:MM:SS)."],
    touch_end_ts: Annotated[Optional[str], "Optional touch end timestamp (YYYY-MM-DD HH:MM:SS)."] = None,
    touch_price_low: Annotated[Optional[float], "Low price during the touch."] = None,
    touch_price_high: Annotated[Optional[float], "High price during the touch."] = None,
    touch_price_close: Annotated[Optional[float], "Close price during the touch."] = None,
    candle_open: Annotated[Optional[float], "Open price of the reference candle."] = None,
    candle_high: Annotated[Optional[float], "High price of the reference candle."] = None,
    candle_low: Annotated[Optional[float], "Low price of the reference candle."] = None,
    candle_close: Annotated[Optional[float], "Close price of the reference candle."] = None,
    candle_volume: Annotated[Optional[int], "Volume of the reference candle."] = None,
    touch_type: Annotated[
        Optional[str],
        "Touch type enum: WICK, BODY, MID, FRONT_RUN, OTHER. Defaults to OTHER.",
    ] = "OTHER",
    zone_status_before: Annotated[
        Optional[str],
        "Zone status before touch: PENDING, ACTIVE, TOUCHED, VIOLATED, INVALIDATED.",
    ] = None,
    zone_status_after: Annotated[
        Optional[str],
        "Zone status after touch: PENDING, ACTIVE, TOUCHED, VIOLATED, INVALIDATED.",
    ] = None,
    reaction_outcome: Annotated[
        Optional[str],
        "Reaction outcome enum: RESPECTED, REJECTED, BROKEN, NO_FILL, OTHER.",
    ] = None,
    reaction_comment: Annotated[Optional[str], "Optional reaction comment."] = None,
) -> dict[str, Any]:
    """Insert a touch event for a zone into ai_zone_touches."""

    repo = _get_zone_touch_repo()
    payload = {
        "zone_id": zone_id,
        "symbol": symbol,
        "timeframe": timeframe,
        "touch_number": touch_number,
        "touch_start_ts": touch_start_ts,
        "touch_end_ts": touch_end_ts,
        "touch_price_low": touch_price_low,
        "touch_price_high": touch_price_high,
        "touch_price_close": touch_price_close,
        "candle_open": candle_open,
        "candle_high": candle_high,
        "candle_low": candle_low,
        "candle_close": candle_close,
        "candle_volume": candle_volume,
        "touch_type": touch_type,
        "zone_status_before": zone_status_before,
        "zone_status_after": zone_status_after,
        "reaction_outcome": reaction_outcome,
        "reaction_comment": reaction_comment,
    }
    touch_id = repo.insert_touch(payload)
    return {"touch_id": touch_id, "status": "created"}


@tool
def add_zone_touch_price_action(
    symbol: Annotated[str, "Symbol identifier that matches the zone touch."],
    timeframe: Annotated[str, "Timeframe enum value: 1m,3m,5m,10m,15m,30m,1H."],
    zone_touch_id: Annotated[int, "Foreign key to ai_zone_touches.id for the referenced touch."],
    pattern_type: Annotated[
        str,
        "Pattern enum: DOUBLE_TOP, DOUBLE_BOTTOM, V_TOP, V_BOTTOM, 20_EMA, OTHER. Defaults to OTHER.",
    ] = "OTHER",
    zone_id: Annotated[Optional[int], "Optional ai_zones.id if available."] = None,
    double_top_id: Annotated[
        Optional[int],
        "Optional ai_double_top_patterns.id if this touch ties to a double top pattern.",
    ] = None,
    double_bottom_id: Annotated[
        Optional[int],
        "Optional ai_double_bottom_patterns.id if this touch ties to a double bottom pattern.",
    ] = None,
    v_top_id: Annotated[Optional[int], "Optional reference to a V top pattern id."] = None,
    v_bottom_id: Annotated[Optional[int], "Optional reference to a V bottom pattern id."] = None,
    ema_20_id: Annotated[Optional[int], "Optional reference to a 20 EMA pattern id."] = None,
    max_favorable_rr: Annotated[
        Optional[float],
        "Optional max favorable risk-reward ratio observed for the touch.",
    ] = None,
) -> dict[str, Any]:
    """Insert a price action record for a zone touch into ai_zone_touch_price_actions."""

    repo = _get_zone_touch_price_action_repo()
    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "zone_id": zone_id,
        "zone_touch_id": zone_touch_id,
        "pattern_type": pattern_type,
        "double_top_id": double_top_id,
        "double_bottom_id": double_bottom_id,
        "v_top_id": v_top_id,
        "v_bottom_id": v_bottom_id,
        "ema_20_id": ema_20_id,
        "max_favorable_rr": max_favorable_rr,
    }
    action_id = repo.insert_action(payload)
    return {"touch_price_action_id": action_id, "status": "created"}


@tool
def update_zone_touch_price_action(
    touch_price_action_id: Annotated[int, "Existing ai_zone_touch_price_actions.id to update."],
    zone_id: Annotated[Optional[int], "Optional ai_zones.id if available."] = None,
    pattern_type: Annotated[
        Optional[str],
        "Pattern enum: DOUBLE_TOP, DOUBLE_BOTTOM, V_TOP, V_BOTTOM, 20_EMA, OTHER.",
    ] = None,
    double_top_id: Annotated[
        Optional[int],
        "Optional ai_double_top_patterns.id if this touch ties to a double top pattern.",
    ] = None,
    double_bottom_id: Annotated[
        Optional[int],
        "Optional ai_double_bottom_patterns.id if this touch ties to a double bottom pattern.",
    ] = None,
    v_top_id: Annotated[Optional[int], "Optional reference to a V top pattern id."] = None,
    v_bottom_id: Annotated[Optional[int], "Optional reference to a V bottom pattern id."] = None,
    ema_20_id: Annotated[Optional[int], "Optional reference to a 20 EMA pattern id."] = None,
    max_favorable_rr: Annotated[
        Optional[float],
        "Optional max favorable risk-reward ratio observed for the touch.",
    ] = None,
) -> dict[str, Any]:
    """Update selected fields on an existing ai_zone_touch_price_actions row."""

    repo = _get_zone_touch_price_action_repo()
    updates = {
        "zone_id": zone_id,
        "pattern_type": pattern_type,
        "double_top_id": double_top_id,
        "double_bottom_id": double_bottom_id,
        "v_top_id": v_top_id,
        "v_bottom_id": v_bottom_id,
        "ema_20_id": ema_20_id,
        "max_favorable_rr": max_favorable_rr,
    }
    rows = repo.update_action(touch_price_action_id, updates)
    return {"touch_price_action_id": touch_price_action_id, "rows_updated": rows}


@tool
def add_double_top_pattern(
    symbol: Annotated[str, "Symbol identifier that matches the pattern."],
    timeframe: Annotated[str, "Timeframe enum: 1m,3m,5m,10m,15m,30m,1H."],
    peak1_timestamp: Annotated[str, "Timestamp of first peak (YYYY-MM-DD HH:MM:SS)."],
    peak2_timestamp: Annotated[str, "Timestamp of second peak (YYYY-MM-DD HH:MM:SS)."],
    peak1_price: Annotated[float, "Price at the first peak."],
    peak2_price: Annotated[float, "Price at the second peak."],
    zone_id: Annotated[Optional[int], "Optional ai_zones.id associated with the pattern."] = None,
    neckline_timestamp: Annotated[Optional[str], "Optional neckline timestamp (YYYY-MM-DD HH:MM:SS)."] = None,
    confirm_timestamp: Annotated[Optional[str], "Optional confirmation timestamp (YYYY-MM-DD HH:MM:SS)."] = None,
    neckline_price: Annotated[Optional[float], "Optional neckline price."] = None,
    peak_diff_abs: Annotated[Optional[float], "Absolute difference between peaks."] = None,
    peak_diff_pct: Annotated[Optional[float], "Percentage difference between peaks."] = None,
    candles_between_peaks: Annotated[Optional[int], "Number of candles between peaks."] = None,
    seconds_between_peaks: Annotated[Optional[int], "Seconds between peaks."] = None,
    cvd_peak1: Annotated[Optional[float], "CVD value at first peak."] = None,
    cvd_peak2: Annotated[Optional[float], "CVD value at second peak."] = None,
    cvd_divergence_side: Annotated[
        str,
        "CVD divergence enum: NONE, BULLISH, BEARISH. Defaults to NONE.",
    ] = "NONE",
    imbalance_side: Annotated[str, "Imbalance enum: NONE, BUY, SELL. Defaults to NONE."] = "NONE",
    imbalance_value: Annotated[Optional[float], "Imbalance metric value."] = None,
    ema20_peak1_position: Annotated[
        Optional[str],
        "EMA20 position at peak1: ABOVE, BELOW, TOUCHING.",
    ] = None,
    ema20_peak2_position: Annotated[
        Optional[str],
        "EMA20 position at peak2: ABOVE, BELOW, TOUCHING.",
    ] = None,
    ema20_peak2_slope: Annotated[Optional[str], "EMA20 slope at peak2: RISING, FALLING, FLAT."] = None,
    sweep_peak2: Annotated[Optional[bool], "Whether peak2 swept prior highs."] = None,
    stop_run_above_highs: Annotated[Optional[bool], "Whether stops were run above highs."] = None,
    quality_score: Annotated[Optional[int], "Optional quality score (0-255)."] = None,
    notes: Annotated[Optional[str], "Optional notes."] = None,
) -> dict[str, Any]:
    """Insert a double top pattern row into ai_double_top_patterns."""

    repo = _get_double_top_pattern_repo()
    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "zone_id": zone_id,
        "peak1_timestamp": peak1_timestamp,
        "peak2_timestamp": peak2_timestamp,
        "neckline_timestamp": neckline_timestamp,
        "confirm_timestamp": confirm_timestamp,
        "peak1_price": peak1_price,
        "peak2_price": peak2_price,
        "neckline_price": neckline_price,
        "peak_diff_abs": peak_diff_abs,
        "peak_diff_pct": peak_diff_pct,
        "candles_between_peaks": candles_between_peaks,
        "seconds_between_peaks": seconds_between_peaks,
        "cvd_peak1": cvd_peak1,
        "cvd_peak2": cvd_peak2,
        "cvd_divergence_side": cvd_divergence_side,
        "imbalance_side": imbalance_side,
        "imbalance_value": imbalance_value,
        "ema20_peak1_position": ema20_peak1_position,
        "ema20_peak2_position": ema20_peak2_position,
        "ema20_peak2_slope": ema20_peak2_slope,
        "sweep_peak2": _coerce_bool(sweep_peak2),
        "stop_run_above_highs": _coerce_bool(stop_run_above_highs),
        "quality_score": quality_score,
        "notes": notes,
    }
    pattern_id = repo.insert_pattern(payload)
    return {"double_top_pattern_id": pattern_id, "status": "created"}


@tool
def update_double_top_pattern(
    pattern_id: Annotated[int, "Existing ai_double_top_patterns.id to update."],
    zone_id: Annotated[Optional[int], "Optional ai_zones.id associated with the pattern."] = None,
    neckline_timestamp: Annotated[Optional[str], "Optional neckline timestamp (YYYY-MM-DD HH:MM:SS)."] = None,
    confirm_timestamp: Annotated[Optional[str], "Optional confirmation timestamp (YYYY-MM-DD HH:MM:SS)."] = None,
    neckline_price: Annotated[Optional[float], "Optional neckline price."] = None,
    peak_diff_abs: Annotated[Optional[float], "Absolute difference between peaks."] = None,
    peak_diff_pct: Annotated[Optional[float], "Percentage difference between peaks."] = None,
    candles_between_peaks: Annotated[Optional[int], "Number of candles between peaks."] = None,
    seconds_between_peaks: Annotated[Optional[int], "Seconds between peaks."] = None,
    cvd_peak1: Annotated[Optional[float], "CVD value at first peak."] = None,
    cvd_peak2: Annotated[Optional[float], "CVD value at second peak."] = None,
    cvd_divergence_side: Annotated[
        Optional[str],
        "CVD divergence enum: NONE, BULLISH, BEARISH.",
    ] = None,
    imbalance_side: Annotated[Optional[str], "Imbalance enum: NONE, BUY, SELL."] = None,
    imbalance_value: Annotated[Optional[float], "Imbalance metric value."] = None,
    ema20_peak1_position: Annotated[
        Optional[str],
        "EMA20 position at peak1: ABOVE, BELOW, TOUCHING.",
    ] = None,
    ema20_peak2_position: Annotated[
        Optional[str],
        "EMA20 position at peak2: ABOVE, BELOW, TOUCHING.",
    ] = None,
    ema20_peak2_slope: Annotated[Optional[str], "EMA20 slope at peak2: RISING, FALLING, FLAT."] = None,
    sweep_peak2: Annotated[Optional[bool], "Whether peak2 swept prior highs."] = None,
    stop_run_above_highs: Annotated[Optional[bool], "Whether stops were run above highs."] = None,
    quality_score: Annotated[Optional[int], "Optional quality score (0-255)."] = None,
    notes: Annotated[Optional[str], "Optional notes."] = None,
) -> dict[str, Any]:
    """Update selected fields for an existing double top pattern."""

    repo = _get_double_top_pattern_repo()
    updates = {
        "zone_id": zone_id,
        "neckline_timestamp": neckline_timestamp,
        "confirm_timestamp": confirm_timestamp,
        "neckline_price": neckline_price,
        "peak_diff_abs": peak_diff_abs,
        "peak_diff_pct": peak_diff_pct,
        "candles_between_peaks": candles_between_peaks,
        "seconds_between_peaks": seconds_between_peaks,
        "cvd_peak1": cvd_peak1,
        "cvd_peak2": cvd_peak2,
        "cvd_divergence_side": cvd_divergence_side,
        "imbalance_side": imbalance_side,
        "imbalance_value": imbalance_value,
        "ema20_peak1_position": ema20_peak1_position,
        "ema20_peak2_position": ema20_peak2_position,
        "ema20_peak2_slope": ema20_peak2_slope,
        "sweep_peak2": _coerce_bool(sweep_peak2),
        "stop_run_above_highs": _coerce_bool(stop_run_above_highs),
        "quality_score": quality_score,
        "notes": notes,
    }
    rows = repo.update_pattern(pattern_id, updates)
    return {"double_top_pattern_id": pattern_id, "rows_updated": rows}


@tool
def add_double_bottom_pattern(
    symbol: Annotated[str, "Symbol identifier that matches the pattern."],
    timeframe: Annotated[str, "Timeframe enum: 1m,3m,5m,10m,15m,30m,1H."],
    bottom1_timestamp: Annotated[str, "Timestamp of first bottom (YYYY-MM-DD HH:MM:SS)."],
    bottom2_timestamp: Annotated[str, "Timestamp of second bottom (YYYY-MM-DD HH:MM:SS)."],
    bottom1_price: Annotated[float, "Price at the first bottom."],
    bottom2_price: Annotated[float, "Price at the second bottom."],
    zone_id: Annotated[Optional[int], "Optional ai_zones.id associated with the pattern."] = None,
    neckline_timestamp: Annotated[Optional[str], "Optional neckline timestamp (YYYY-MM-DD HH:MM:SS)."] = None,
    confirm_timestamp: Annotated[Optional[str], "Optional confirmation timestamp (YYYY-MM-DD HH:MM:SS)."] = None,
    neckline_price: Annotated[Optional[float], "Optional neckline price."] = None,
    bottom_diff_abs: Annotated[Optional[float], "Absolute difference between bottoms."] = None,
    bottom_diff_pct: Annotated[Optional[float], "Percentage difference between bottoms."] = None,
    candles_between_bottoms: Annotated[Optional[int], "Number of candles between bottoms."] = None,
    seconds_between_bottoms: Annotated[Optional[int], "Seconds between bottoms."] = None,
    cvd_bottom1: Annotated[Optional[float], "CVD value at first bottom."] = None,
    cvd_bottom2: Annotated[Optional[float], "CVD value at second bottom."] = None,
    cvd_divergence_side: Annotated[
        str,
        "CVD divergence enum: NONE, BULLISH, BEARISH. Defaults to NONE.",
    ] = "NONE",
    imbalance_side: Annotated[str, "Imbalance enum: NONE, BUY, SELL. Defaults to NONE."] = "NONE",
    imbalance_value: Annotated[Optional[float], "Imbalance metric value."] = None,
    ema20_bottom1_position: Annotated[
        Optional[str],
        "EMA20 position at bottom1: ABOVE, BELOW, TOUCHING.",
    ] = None,
    ema20_bottom2_position: Annotated[
        Optional[str],
        "EMA20 position at bottom2: ABOVE, BELOW, TOUCHING.",
    ] = None,
    ema20_bottom2_slope: Annotated[Optional[str], "EMA20 slope at bottom2: RISING, FALLING, FLAT."] = None,
    sweep_bottom2: Annotated[Optional[bool], "Whether bottom2 swept prior lows."] = None,
    stop_run_below_lows: Annotated[Optional[bool], "Whether stops were run below lows."] = None,
    quality_score: Annotated[Optional[int], "Optional quality score (0-255)."] = None,
    notes: Annotated[Optional[str], "Optional notes."] = None,
) -> dict[str, Any]:
    """Insert a double bottom pattern row into ai_double_bottom_patterns."""

    repo = _get_double_bottom_pattern_repo()
    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "zone_id": zone_id,
        "bottom1_timestamp": bottom1_timestamp,
        "bottom2_timestamp": bottom2_timestamp,
        "neckline_timestamp": neckline_timestamp,
        "confirm_timestamp": confirm_timestamp,
        "bottom1_price": bottom1_price,
        "bottom2_price": bottom2_price,
        "neckline_price": neckline_price,
        "bottom_diff_abs": bottom_diff_abs,
        "bottom_diff_pct": bottom_diff_pct,
        "candles_between_bottoms": candles_between_bottoms,
        "seconds_between_bottoms": seconds_between_bottoms,
        "cvd_bottom1": cvd_bottom1,
        "cvd_bottom2": cvd_bottom2,
        "cvd_divergence_side": cvd_divergence_side,
        "imbalance_side": imbalance_side,
        "imbalance_value": imbalance_value,
        "ema20_bottom1_position": ema20_bottom1_position,
        "ema20_bottom2_position": ema20_bottom2_position,
        "ema20_bottom2_slope": ema20_bottom2_slope,
        "sweep_bottom2": _coerce_bool(sweep_bottom2),
        "stop_run_below_lows": _coerce_bool(stop_run_below_lows),
        "quality_score": quality_score,
        "notes": notes,
    }
    pattern_id = repo.insert_pattern(payload)
    return {"double_bottom_pattern_id": pattern_id, "status": "created"}


@tool
def update_double_bottom_pattern(
    pattern_id: Annotated[int, "Existing ai_double_bottom_patterns.id to update."],
    zone_id: Annotated[Optional[int], "Optional ai_zones.id associated with the pattern."] = None,
    neckline_timestamp: Annotated[Optional[str], "Optional neckline timestamp (YYYY-MM-DD HH:MM:SS)."] = None,
    confirm_timestamp: Annotated[Optional[str], "Optional confirmation timestamp (YYYY-MM-DD HH:MM:SS)."] = None,
    neckline_price: Annotated[Optional[float], "Optional neckline price."] = None,
    bottom_diff_abs: Annotated[Optional[float], "Absolute difference between bottoms."] = None,
    bottom_diff_pct: Annotated[Optional[float], "Percentage difference between bottoms."] = None,
    candles_between_bottoms: Annotated[Optional[int], "Number of candles between bottoms."] = None,
    seconds_between_bottoms: Annotated[Optional[int], "Seconds between bottoms."] = None,
    cvd_bottom1: Annotated[Optional[float], "CVD value at first bottom."] = None,
    cvd_bottom2: Annotated[Optional[float], "CVD value at second bottom."] = None,
    cvd_divergence_side: Annotated[
        Optional[str],
        "CVD divergence enum: NONE, BULLISH, BEARISH.",
    ] = None,
    imbalance_side: Annotated[Optional[str], "Imbalance enum: NONE, BUY, SELL."] = None,
    imbalance_value: Annotated[Optional[float], "Imbalance metric value."] = None,
    ema20_bottom1_position: Annotated[
        Optional[str],
        "EMA20 position at bottom1: ABOVE, BELOW, TOUCHING.",
    ] = None,
    ema20_bottom2_position: Annotated[
        Optional[str],
        "EMA20 position at bottom2: ABOVE, BELOW, TOUCHING.",
    ] = None,
    ema20_bottom2_slope: Annotated[Optional[str], "EMA20 slope at bottom2: RISING, FALLING, FLAT."] = None,
    sweep_bottom2: Annotated[Optional[bool], "Whether bottom2 swept prior lows."] = None,
    stop_run_below_lows: Annotated[Optional[bool], "Whether stops were run below lows."] = None,
    quality_score: Annotated[Optional[int], "Optional quality score (0-255)."] = None,
    notes: Annotated[Optional[str], "Optional notes."] = None,
) -> dict[str, Any]:
    """Update selected fields for an existing double bottom pattern."""

    repo = _get_double_bottom_pattern_repo()
    updates = {
        "zone_id": zone_id,
        "neckline_timestamp": neckline_timestamp,
        "confirm_timestamp": confirm_timestamp,
        "neckline_price": neckline_price,
        "bottom_diff_abs": bottom_diff_abs,
        "bottom_diff_pct": bottom_diff_pct,
        "candles_between_bottoms": candles_between_bottoms,
        "seconds_between_bottoms": seconds_between_bottoms,
        "cvd_bottom1": cvd_bottom1,
        "cvd_bottom2": cvd_bottom2,
        "cvd_divergence_side": cvd_divergence_side,
        "imbalance_side": imbalance_side,
        "imbalance_value": imbalance_value,
        "ema20_bottom1_position": ema20_bottom1_position,
        "ema20_bottom2_position": ema20_bottom2_position,
        "ema20_bottom2_slope": ema20_bottom2_slope,
        "sweep_bottom2": _coerce_bool(sweep_bottom2),
        "stop_run_below_lows": _coerce_bool(stop_run_below_lows),
        "quality_score": quality_score,
        "notes": notes,
    }
    rows = repo.update_pattern(pattern_id, updates)
    return {"double_bottom_pattern_id": pattern_id, "rows_updated": rows}


@tool
def add_v_top_pattern(
    symbol: Annotated[str, "Symbol identifier that matches the pattern."],
    timeframe: Annotated[str, "Timeframe enum: 1m,3m,5m,10m,15m,30m,1H."],
    peak_timestamp: Annotated[str, "Timestamp of the V peak (YYYY-MM-DD HH:MM:SS)."],
    peak_price: Annotated[float, "Price at the peak."],
    rejection_timestamp: Annotated[str, "Timestamp of rejection (YYYY-MM-DD HH:MM:SS)."],
    rejection_price_low: Annotated[float, "Lowest price during rejection leg."],
    zone_id: Annotated[Optional[int], "Optional ai_zones.id associated with the pattern."] = None,
    drop_abs: Annotated[Optional[float], "Absolute price drop from peak to rejection low."] = None,
    drop_pct: Annotated[Optional[float], "Percentage price drop from peak to rejection low."] = None,
    seconds_to_drop: Annotated[Optional[int], "Seconds from peak to rejection low."] = None,
    candles_to_drop: Annotated[Optional[int], "Candles from peak to rejection low."] = None,
    sweep_peak: Annotated[Optional[bool], "Whether the peak swept prior highs."] = None,
    stop_run_above_highs: Annotated[Optional[bool], "Whether stops were run above highs."] = None,
    cvd_peak: Annotated[Optional[float], "CVD value at the peak."] = None,
    cvd_after_drop: Annotated[Optional[float], "CVD value after the drop."] = None,
    cvd_shift_pct: Annotated[Optional[float], "Percentage shift in CVD across the drop."] = None,
    imbalance_side: Annotated[str, "Imbalance enum: NONE, BUY, SELL. Defaults to NONE."] = "NONE",
    imbalance_value: Annotated[Optional[float], "Imbalance metric value."] = None,
    ema20_peak_position: Annotated[
        Optional[str],
        "EMA20 position at peak: ABOVE, BELOW, TOUCHING.",
    ] = None,
    ema20_peak_slope: Annotated[Optional[str], "EMA20 slope at peak: RISING, FALLING, FLAT."] = None,
    quality_score: Annotated[Optional[int], "Optional quality score (0-255)."] = None,
    notes: Annotated[Optional[str], "Optional notes."] = None,
) -> dict[str, Any]:
    """Insert a V top pattern row into ai_v_top_patterns."""

    repo = _get_v_top_pattern_repo()
    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "zone_id": zone_id,
        "peak_timestamp": peak_timestamp,
        "peak_price": peak_price,
        "rejection_timestamp": rejection_timestamp,
        "rejection_price_low": rejection_price_low,
        "drop_abs": drop_abs,
        "drop_pct": drop_pct,
        "seconds_to_drop": seconds_to_drop,
        "candles_to_drop": candles_to_drop,
        "sweep_peak": _coerce_bool(sweep_peak),
        "stop_run_above_highs": _coerce_bool(stop_run_above_highs),
        "cvd_peak": cvd_peak,
        "cvd_after_drop": cvd_after_drop,
        "cvd_shift_pct": cvd_shift_pct,
        "imbalance_side": imbalance_side,
        "imbalance_value": imbalance_value,
        "ema20_peak_position": ema20_peak_position,
        "ema20_peak_slope": ema20_peak_slope,
        "quality_score": quality_score,
        "notes": notes,
    }
    pattern_id = repo.insert_pattern(payload)
    return {"v_top_pattern_id": pattern_id, "status": "created"}


@tool
def update_v_top_pattern(
    pattern_id: Annotated[int, "Existing ai_v_top_patterns.id to update."],
    zone_id: Annotated[Optional[int], "Optional ai_zones.id associated with the pattern."] = None,
    rejection_timestamp: Annotated[Optional[str], "Optional rejection timestamp (YYYY-MM-DD HH:MM:SS)."] = None,
    rejection_price_low: Annotated[Optional[float], "Optional lowest price during rejection leg."] = None,
    drop_abs: Annotated[Optional[float], "Absolute price drop from peak to rejection low."] = None,
    drop_pct: Annotated[Optional[float], "Percentage price drop from peak to rejection low."] = None,
    seconds_to_drop: Annotated[Optional[int], "Seconds from peak to rejection low."] = None,
    candles_to_drop: Annotated[Optional[int], "Candles from peak to rejection low."] = None,
    sweep_peak: Annotated[Optional[bool], "Whether the peak swept prior highs."] = None,
    stop_run_above_highs: Annotated[Optional[bool], "Whether stops were run above highs."] = None,
    cvd_peak: Annotated[Optional[float], "CVD value at the peak."] = None,
    cvd_after_drop: Annotated[Optional[float], "CVD value after the drop."] = None,
    cvd_shift_pct: Annotated[Optional[float], "Percentage shift in CVD across the drop."] = None,
    imbalance_side: Annotated[Optional[str], "Imbalance enum: NONE, BUY, SELL."] = None,
    imbalance_value: Annotated[Optional[float], "Imbalance metric value."] = None,
    ema20_peak_position: Annotated[
        Optional[str],
        "EMA20 position at peak: ABOVE, BELOW, TOUCHING.",
    ] = None,
    ema20_peak_slope: Annotated[Optional[str], "EMA20 slope at peak: RISING, FALLING, FLAT."] = None,
    quality_score: Annotated[Optional[int], "Optional quality score (0-255)."] = None,
    notes: Annotated[Optional[str], "Optional notes."] = None,
) -> dict[str, Any]:
    """Update selected fields for an existing V top pattern."""

    repo = _get_v_top_pattern_repo()
    updates = {
        "zone_id": zone_id,
        "rejection_timestamp": rejection_timestamp,
        "rejection_price_low": rejection_price_low,
        "drop_abs": drop_abs,
        "drop_pct": drop_pct,
        "seconds_to_drop": seconds_to_drop,
        "candles_to_drop": candles_to_drop,
        "sweep_peak": _coerce_bool(sweep_peak),
        "stop_run_above_highs": _coerce_bool(stop_run_above_highs),
        "cvd_peak": cvd_peak,
        "cvd_after_drop": cvd_after_drop,
        "cvd_shift_pct": cvd_shift_pct,
        "imbalance_side": imbalance_side,
        "imbalance_value": imbalance_value,
        "ema20_peak_position": ema20_peak_position,
        "ema20_peak_slope": ema20_peak_slope,
        "quality_score": quality_score,
        "notes": notes,
    }
    rows = repo.update_pattern(pattern_id, updates)
    return {"v_top_pattern_id": pattern_id, "rows_updated": rows}


@tool
def add_ema_20_pattern(
    symbol: Annotated[str, "Symbol identifier that matches the pattern."],
    timeframe: Annotated[str, "Timeframe enum: 1m,3m,5m,10m,15m,30m,1H."],
    pattern_timestamp: Annotated[str, "Timestamp of the EMA interaction (YYYY-MM-DD HH:MM:SS)."],
    price_position: Annotated[str, "Price position relative to EMA: ABOVE, BELOW, TOUCHING."],
    ema20_slope: Annotated[str, "EMA20 slope at the interaction: RISING, FALLING, FLAT."],
    ema_pattern_type: Annotated[
        str,
        "Pattern enum: EMA_REJECTION, EMA_SUPPORT, EMA_RESISTANCE, EMA_BREAK_AND_RETEST, EMA_MEAN_REVERSION, EMA_CHOP, EMA_TRANSITION, OTHER.",
    ] = "OTHER",
    pattern_start_ts: Annotated[Optional[str], "Optional pattern start timestamp (YYYY-MM-DD HH:MM:SS)."] = None,
    pattern_end_ts: Annotated[Optional[str], "Optional pattern end timestamp (YYYY-MM-DD HH:MM:SS)."] = None,
    rejection_side: Annotated[str, "Rejection side enum: NONE, UP, DOWN. Defaults to NONE."] = "NONE",
    distance_from_ema_abs: Annotated[Optional[float], "Absolute distance from price to EMA."] = None,
    distance_from_ema_pct: Annotated[Optional[float], "Percentage distance from price to EMA."] = None,
    cvd_bias: Annotated[str, "CVD bias enum: BULLISH, BEARISH, NEUTRAL, UNKNOWN. Defaults to UNKNOWN."] = "UNKNOWN",
    imbalance_side: Annotated[str, "Imbalance enum: NONE, BUY, SELL. Defaults to NONE."] = "NONE",
    imbalance_value: Annotated[Optional[float], "Order flow imbalance metric."] = None,
    notes: Annotated[Optional[str], "Optional notes for the pattern."] = None,
) -> dict[str, Any]:
    """Insert a 20 EMA pattern row into ai_ema_20_patterns."""

    repo = _get_ema20_pattern_repo()
    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "ema_pattern_type": ema_pattern_type,
        "pattern_timestamp": pattern_timestamp,
        "pattern_start_ts": pattern_start_ts,
        "pattern_end_ts": pattern_end_ts,
        "price_position": price_position,
        "rejection_side": rejection_side,
        "ema20_slope": ema20_slope,
        "distance_from_ema_abs": distance_from_ema_abs,
        "distance_from_ema_pct": distance_from_ema_pct,
        "cvd_bias": cvd_bias,
        "imbalance_side": imbalance_side,
        "imbalance_value": imbalance_value,
        "notes": notes,
    }
    pattern_id = repo.insert_pattern(payload)
    return {"ema_20_pattern_id": pattern_id, "status": "created"}


@tool
def update_ema_20_pattern(
    ema_20_pattern_id: Annotated[int, "Existing ai_ema_20_patterns.id to update."],
    pattern_timestamp: Annotated[Optional[str], "Optional EMA interaction timestamp (YYYY-MM-DD HH:MM:SS)."] = None,
    pattern_start_ts: Annotated[Optional[str], "Optional pattern start timestamp (YYYY-MM-DD HH:MM:SS)."] = None,
    pattern_end_ts: Annotated[Optional[str], "Optional pattern end timestamp (YYYY-MM-DD HH:MM:SS)."] = None,
    price_position: Annotated[Optional[str], "Price position relative to EMA: ABOVE, BELOW, TOUCHING."] = None,
    rejection_side: Annotated[Optional[str], "Rejection side enum: NONE, UP, DOWN."] = None,
    ema20_slope: Annotated[Optional[str], "EMA20 slope at the interaction: RISING, FALLING, FLAT."] = None,
    ema_pattern_type: Annotated[
        Optional[str],
        "Pattern enum: EMA_REJECTION, EMA_SUPPORT, EMA_RESISTANCE, EMA_BREAK_AND_RETEST, EMA_MEAN_REVERSION, EMA_CHOP, EMA_TRANSITION, OTHER.",
    ] = None,
    distance_from_ema_abs: Annotated[Optional[float], "Absolute distance from price to EMA."] = None,
    distance_from_ema_pct: Annotated[Optional[float], "Percentage distance from price to EMA."] = None,
    cvd_bias: Annotated[Optional[str], "CVD bias enum: BULLISH, BEARISH, NEUTRAL, UNKNOWN."] = None,
    imbalance_side: Annotated[Optional[str], "Imbalance enum: NONE, BUY, SELL."] = None,
    imbalance_value: Annotated[Optional[float], "Order flow imbalance metric."] = None,
    notes: Annotated[Optional[str], "Optional notes for the pattern."] = None,
) -> dict[str, Any]:
    """Update selected fields on an existing ai_ema_20_patterns row."""

    repo = _get_ema20_pattern_repo()
    updates = {
        "pattern_timestamp": pattern_timestamp,
        "pattern_start_ts": pattern_start_ts,
        "pattern_end_ts": pattern_end_ts,
        "price_position": price_position,
        "rejection_side": rejection_side,
        "ema20_slope": ema20_slope,
        "ema_pattern_type": ema_pattern_type,
        "distance_from_ema_abs": distance_from_ema_abs,
        "distance_from_ema_pct": distance_from_ema_pct,
        "cvd_bias": cvd_bias,
        "imbalance_side": imbalance_side,
        "imbalance_value": imbalance_value,
        "notes": notes,
    }
    rows = repo.update_pattern(ema_20_pattern_id, updates)
    return {"ema_20_pattern_id": ema_20_pattern_id, "rows_updated": rows}


@tool
def add_zone_relationship(
    child_zone_id: Annotated[int, "Child ai_zones.id that references a parent zone."],
    parent_zone_id: Annotated[int, "Parent ai_zones.id that the child depends on."],
    relationship_type: Annotated[
        Optional[str],
        "Relationship enum: DIRECT_PARENT, CONFLUENCE, FLIP_REFERENCE. Defaults to DIRECT_PARENT.",
    ] = "DIRECT_PARENT",
    confidence_impact: Annotated[
        Optional[str],
        "Confidence impact enum: LOW, MEDIUM, HIGH. Defaults to HIGH.",
    ] = "HIGH",
) -> dict[str, Any]:
    """Insert a parent/child relationship between zones into ai_zone_relationships."""

    repo = _get_zone_relationship_repo()
    payload = {
        "child_zone_id": child_zone_id,
        "parent_zone_id": parent_zone_id,
        "relationship_type": relationship_type,
        "confidence_impact": confidence_impact,
    }
    relationship_id = repo.insert_relationship(payload)
    return {"relationship_id": relationship_id, "status": "created"}


TRADING_TOOLS = [get_candles, get_ema, get_current_date, compare_dates]
ZONE_TOOLS = [
    create_zone,
    update_zone,
    get_zones,
    add_report_note,
    get_report_notes,
    update_report_note,
    add_zone_note,
    add_zone_touch,
    add_zone_touch_price_action,
    update_zone_touch_price_action,
    add_double_top_pattern,
    update_double_top_pattern,
    add_double_bottom_pattern,
    update_double_bottom_pattern,
    add_v_top_pattern,
    update_v_top_pattern,
    add_ema_20_pattern,
    update_ema_20_pattern,
    add_zone_relationship,
]
MARKET_TOOLS = [*TRADING_TOOLS, *ZONE_TOOLS]

__all__ = [
    "TRADING_TOOLS",
    "ZONE_TOOLS",
    "MARKET_TOOLS",
    "get_candles",
    "get_ema",
    "get_current_date",
    "compare_dates",
    "create_zone",
    "update_zone",
    "get_zones",
    "add_report_note",
    "get_report_notes",
    "update_report_note",
    "ZoneRepository",
    "ZoneNoteRepository",
    "ZoneTouchRepository",
    "ZoneTouchPriceActionRepository",
    "DoubleTopPatternRepository",
    "DoubleBottomPatternRepository",
    "VTopPatternRepository",
    "Ema20PatternRepository",
    "ZoneRelationshipRepository",
    "add_zone_note",
    "add_zone_touch",
    "add_zone_touch_price_action",
    "update_zone_touch_price_action",
    "add_double_top_pattern",
    "update_double_top_pattern",
    "add_double_bottom_pattern",
    "update_double_bottom_pattern",
    "add_v_top_pattern",
    "update_v_top_pattern",
    "add_ema_20_pattern",
    "update_ema_20_pattern",
    "add_zone_relationship",
]
