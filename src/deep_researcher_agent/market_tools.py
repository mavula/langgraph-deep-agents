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
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        status: Optional[str] = None,
        zone_type: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        sql_parts = [f"SELECT * FROM {self._table} WHERE 1=1"]
        params: list[Any] = []

        if symbol:
            sql_parts.append("AND symbol = %s")
            params.append(symbol)
        if timeframe:
            sql_parts.append("AND timeframe = %s")
            params.append(timeframe)
        if status:
            sql_parts.append("AND status = %s")
            params.append(status)
        if zone_type:
            sql_parts.append("AND zone_type = %s")
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
    cvd_confirmation_type: Annotated[Optional[str], "CVD confirmation enum: NONE, RISING_AWAY, DIVERGENCE_RETEST."] = "NONE",
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
        "cvd_comment": cvd_comment,
    }
    rows = repo.update_zone(zone_id, updates)
    return {"zone_id": zone_id, "rows_updated": rows}


@tool
def get_zones(
    symbol: Annotated[Optional[str], "Optional symbol filter."] = None,
    timeframe: Annotated[Optional[str], "Optional timeframe filter."] = None,
    status: Annotated[Optional[str], "Optional status filter."] = None,
    zone_type: Annotated[Optional[str], "Optional zone type filter: DEMAND or SUPPLY."] = None,
    limit: Annotated[int, "Maximum rows to return (1-500)."] = 50,
) -> list[dict]:
    """Fetch zones from ai_zones with optional filters."""

    repo = _get_zone_repo()
    return repo.fetch_zones(symbol=symbol, timeframe=timeframe, status=status, zone_type=zone_type, limit=limit)


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
    add_zone_note,
    add_zone_touch,
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
    "ZoneRepository",
    "ZoneNoteRepository",
    "ZoneTouchRepository",
    "ZoneRelationshipRepository",
    "add_zone_note",
    "add_zone_touch",
    "add_zone_relationship",
]
