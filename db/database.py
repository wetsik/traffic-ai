from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

from config import DB_PATH


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=NORMAL')
    conn.execute('PRAGMA foreign_keys=ON')
    conn.execute('PRAGMA busy_timeout=5000')
    _ensure_schema(conn)
    try:
        yield conn
    finally:
        conn.close()


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        '''
        CREATE TABLE IF NOT EXISTS detections_raw (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            pedestrians INTEGER NOT NULL DEFAULT 0,
            cars INTEGER NOT NULL DEFAULT 0,
            vehicles INTEGER NOT NULL DEFAULT 0
        )
        '''
    )
    conn.execute('CREATE INDEX IF NOT EXISTS idx_detections_raw_timestamp ON detections_raw(timestamp)')
    conn.execute(
        '''
        CREATE TABLE IF NOT EXISTS daily_summary (
            date DATE PRIMARY KEY,
            pedestrians INTEGER NOT NULL DEFAULT 0,
            cars INTEGER NOT NULL DEFAULT 0,
            vehicles INTEGER NOT NULL DEFAULT 0
        )
        '''
    )


def _normalize_timestamp(value: datetime | str | None) -> datetime:
    if value is None:
        return datetime.now()
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _normalize_date(value: date | datetime | str) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return datetime.fromisoformat(str(value)).date()


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    data = dict(row)
    if 'cars' in data and 'vehicles' in data:
        data['combined_vehicles'] = int(data['cars'] or 0) + int(data['vehicles'] or 0)
    return data


def _cleanup_old_records(conn: sqlite3.Connection) -> None:
    cutoff = (datetime.now() - timedelta(hours=3)).isoformat(sep=' ', timespec='seconds')
    conn.execute('DELETE FROM detections_raw WHERE timestamp < ?', (cutoff,))


def cleanup_old_records() -> None:
    with _connect() as conn:
        _cleanup_old_records(conn)
        conn.commit()


def _increment_daily_summary(
    conn: sqlite3.Connection,
    summary_date: date,
    pedestrians: int,
    cars: int,
    vehicles: int,
) -> None:
    conn.execute(
        '''
        INSERT INTO daily_summary (date, pedestrians, cars, vehicles)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(date) DO UPDATE SET
            pedestrians = pedestrians + excluded.pedestrians,
            cars = cars + excluded.cars,
            vehicles = vehicles + excluded.vehicles
        ''',
        (summary_date.isoformat(), pedestrians, cars, vehicles),
    )


def upsert_daily_summary(
    summary_date: date | datetime | str,
    pedestrians: int,
    cars: int,
    vehicles: int,
) -> None:
    day = _normalize_date(summary_date)
    with _connect() as conn:
        conn.execute(
            '''
            INSERT INTO daily_summary (date, pedestrians, cars, vehicles)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                pedestrians = excluded.pedestrians,
                cars = excluded.cars,
                vehicles = excluded.vehicles
            ''',
            (day.isoformat(), pedestrians, cars, vehicles),
        )
        conn.commit()


def insert_detection(
    pedestrians: int,
    cars: int,
    vehicles: int,
    timestamp: datetime | str | None = None,
) -> int:
    ts = _normalize_timestamp(timestamp)
    day = ts.date()
    ts_value = ts.isoformat(sep=' ', timespec='seconds')

    with _connect() as conn:
        cursor = conn.execute(
            '''
            INSERT INTO detections_raw (timestamp, pedestrians, cars, vehicles)
            VALUES (?, ?, ?, ?)
            ''',
            (ts_value, pedestrians, cars, vehicles),
        )
        _increment_daily_summary(conn, day, pedestrians, cars, vehicles)
        _cleanup_old_records(conn)
        conn.commit()
        return int(cursor.lastrowid)


def get_last_3h() -> list[dict[str, Any]]:
    cutoff = (datetime.now() - timedelta(hours=3)).isoformat(sep=' ', timespec='seconds')
    with _connect() as conn:
        rows = conn.execute(
            '''
            SELECT id, timestamp, pedestrians, cars, vehicles
            FROM detections_raw
            WHERE timestamp >= ?
            ORDER BY timestamp ASC, id ASC
            ''',
            (cutoff,),
        ).fetchall()
    return [dict(row) | {'combined_vehicles': int(row['cars']) + int(row['vehicles'])} for row in rows]


def get_latest_detection() -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            '''
            SELECT id, timestamp, pedestrians, cars, vehicles
            FROM detections_raw
            ORDER BY timestamp DESC, id DESC
            LIMIT 1
            '''
        ).fetchone()
    return _row_to_dict(row)


def get_current_count() -> dict[str, Any]:
    """Get the most recent detection record."""
    latest = get_latest_detection()
    if latest:
        return {
            'pedestrians': int(latest.get('pedestrians', 0)),
            'cars': int(latest.get('cars', 0)),
            'vehicles': int(latest.get('vehicles', 0)),
            'timestamp': latest.get('timestamp'),
        }
    return {'pedestrians': 0, 'cars': 0, 'vehicles': 0, 'timestamp': None}


def get_daily_summary(days: int = 30) -> list[dict[str, Any]]:
    days = max(1, int(days))
    cutoff = (date.today() - timedelta(days=days - 1)).isoformat()
    with _connect() as conn:
        rows = conn.execute(
            '''
            SELECT date, pedestrians, cars, vehicles
            FROM daily_summary
            WHERE date >= ?
            ORDER BY date ASC
            ''',
            (cutoff,),
        ).fetchall()
    return [dict(row) | {'combined_vehicles': int(row['cars']) + int(row['vehicles'])} for row in rows]


def get_daily_summary_for_date(summary_date: date | datetime | str) -> dict[str, Any] | None:
    day = _normalize_date(summary_date).isoformat()
    with _connect() as conn:
        row = conn.execute(
            '''
            SELECT date, pedestrians, cars, vehicles
            FROM daily_summary
            WHERE date = ?
            LIMIT 1
            '''
            , (day,),
        ).fetchone()
    return _row_to_dict(row)


if __name__ == '__main__':
    cleanup_old_records()
    print(f'Database ready at {Path(DB_PATH)}')
