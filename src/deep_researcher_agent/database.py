"""Simple database helpers for MySQL-backed tools."""

from __future__ import annotations

import contextlib
from typing import Any, Dict, Iterable, List

try:
    import pymysql  # type: ignore
    import pymysql.cursors  # type: ignore
except ImportError as exc:  # pragma: no cover - optional dependency guard
    pymysql = None  # type: ignore
    _PYMYSQL_IMPORT_ERROR = exc
else:
    _PYMYSQL_IMPORT_ERROR = None

from .config import DatabaseConfig


class DatabaseClient:
    """Tiny wrapper around pymysql to keep connection logic in one place."""

    def __init__(self, config: DatabaseConfig) -> None:
        self._config = config

    @contextlib.contextmanager
    def connection(self) -> Iterable[pymysql.connections.Connection]:
        """Context manager that yields a fresh connection."""

        if _PYMYSQL_IMPORT_ERROR is not None:
            raise RuntimeError(
                "pymysql is required for database tools but is not installed. "
                "Install backend extra dependencies (e.g., `pip install -e .[dev]`)."
            ) from _PYMYSQL_IMPORT_ERROR

        conn = pymysql.connect(
            host=self._config.host,
            port=self._config.port,
            user=self._config.user,
            password=self._config.password,
            database=self._config.database,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True,
        )
        try:
            yield conn
        finally:
            conn.close()

    def fetch_all(self, sql: str, params: Iterable[Any]) -> List[Dict[str, Any]]:
        """Execute the select query and return all rows as dicts."""

        with self.connection() as conn, conn.cursor() as cursor:
            cursor.execute(sql, tuple(params))
            return list(cursor.fetchall())
