"""SQLite persistence for locally accumulated event and calendar data."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .engine import validate_input


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _instant(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"Datetime must include timezone: {value}")
    return parsed.astimezone(timezone.utc)


class Store:
    def __init__(self, path: str | Path = "data/calendar_agent.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                PRAGMA foreign_keys = ON;
                CREATE TABLE IF NOT EXISTS sources (
                    source_id TEXT PRIMARY KEY,
                    data_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    data_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS calendar_events (
                    event_id TEXT PRIMARY KEY,
                    data_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS travel_estimates (
                    from_event_id TEXT NOT NULL,
                    to_event_id TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    PRIMARY KEY (from_event_id, to_event_id)
                );
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS duplicate_reviews (
                    pair_key TEXT PRIMARY KEY,
                    left_id TEXT NOT NULL,
                    right_id TEXT NOT NULL,
                    event_ids_json TEXT NOT NULL,
                    calendar_event_ids_json TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS analysis_runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    result_json TEXT NOT NULL
                );
                """
            )

    def import_document(self, data: dict[str, Any]) -> dict[str, int]:
        validate_input(data)
        counts = {"sources": 0, "events": 0, "calendar_events": 0, "travel_estimates": 0}
        with self._connect() as db:
            for source in data["sources"]:
                db.execute("INSERT INTO sources VALUES (?, ?) ON CONFLICT(source_id) DO UPDATE SET data_json=excluded.data_json", (source["source_id"], _dump(source)))
                counts["sources"] += 1
            for event in data["event_candidates"]:
                db.execute("INSERT INTO events VALUES (?, ?) ON CONFLICT(event_id) DO UPDATE SET data_json=excluded.data_json", (event["event_id"], _dump(event)))
                counts["events"] += 1
            for event in data["calendar_context"]["events"]:
                db.execute("INSERT INTO calendar_events VALUES (?, ?) ON CONFLICT(event_id) DO UPDATE SET data_json=excluded.data_json", (event["event_id"], _dump(event)))
                counts["calendar_events"] += 1
            for estimate in data["calendar_context"].get("travel_estimates", []):
                db.execute(
                    "INSERT INTO travel_estimates VALUES (?, ?, ?) ON CONFLICT(from_event_id, to_event_id) DO UPDATE SET data_json=excluded.data_json",
                    (estimate["from_event_id"], estimate["to_event_id"], _dump(estimate)),
                )
                counts["travel_estimates"] += 1

            old_user_row = db.execute("SELECT value_json FROM settings WHERE key='user'").fetchone()
            old_user = json.loads(old_user_row[0]) if old_user_row else {}
            merged_user = {**old_user, **data["user"]}
            db.execute("INSERT INTO settings VALUES ('user', ?) ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json", (_dump(merged_user),))

            old_start_row = db.execute("SELECT value_json FROM settings WHERE key='range_start'").fetchone()
            old_end_row = db.execute("SELECT value_json FROM settings WHERE key='range_end'").fetchone()
            new_start = data["calendar_context"]["range_start"]
            new_end = data["calendar_context"]["range_end"]
            if old_start_row:
                old_start = json.loads(old_start_row[0])
                new_start = min((old_start, new_start), key=_instant)
            if old_end_row:
                old_end = json.loads(old_end_row[0])
                new_end = max((old_end, new_end), key=_instant)
            for key, value in (("range_start", new_start), ("range_end", new_end)):
                db.execute("INSERT INTO settings VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json", (key, _dump(value)))
        return counts

    def build_input(self, request_id: str) -> dict[str, Any]:
        with self._connect() as db:
            sources = [json.loads(row[0]) for row in db.execute("SELECT data_json FROM sources ORDER BY source_id")]
            events = [json.loads(row[0]) for row in db.execute("SELECT data_json FROM events ORDER BY event_id")]
            calendar_events = [json.loads(row[0]) for row in db.execute("SELECT data_json FROM calendar_events ORDER BY event_id")]
            travel_estimates = [json.loads(row[0]) for row in db.execute("SELECT data_json FROM travel_estimates ORDER BY from_event_id, to_event_id")]
            settings = {row["key"]: json.loads(row["value_json"]) for row in db.execute("SELECT key, value_json FROM settings")}
        if not events:
            raise ValueError("No event candidates are stored yet. Import a JSON file first.")
        if "user" not in settings or "range_start" not in settings or "range_end" not in settings:
            raise ValueError("Database settings are incomplete. Import a full input JSON document first.")
        return {
            "schema_version": "1.0",
            "request_id": request_id,
            "user": settings["user"],
            "sources": sources,
            "event_candidates": events,
            "calendar_context": {
                "range_start": settings["range_start"],
                "range_end": settings["range_end"],
                "events": calendar_events,
                "travel_estimates": travel_estimates,
            },
        }

    def list_records(self, kind: str) -> list[dict[str, Any]]:
        tables = {"sources": "sources", "events": "events", "calendar": "calendar_events"}
        if kind not in tables:
            raise ValueError(f"Unsupported record kind: {kind}")
        with self._connect() as db:
            rows = db.execute(f"SELECT data_json FROM {tables[kind]} ORDER BY 1").fetchall()
        return [json.loads(row[0]) for row in rows]

    def save_duplicate_issues(self, issues: list[dict[str, Any]]) -> int:
        saved = 0
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as db:
            for issue in issues:
                if issue.get("type") != "possible_duplicate":
                    continue
                event_ids = list(issue.get("event_ids", []))
                calendar_ids = list(issue.get("calendar_event_ids", []))
                pairs: list[tuple[str, str]] = []
                for event_id in event_ids:
                    pairs.extend((event_id, calendar_id) for calendar_id in calendar_ids)
                if len(event_ids) > 1:
                    pairs.extend((event_ids[i], event_ids[j]) for i in range(len(event_ids)) for j in range(i + 1, len(event_ids)))
                for left_id, right_id in pairs:
                    ordered = sorted((left_id, right_id))
                    pair_key = _dump(ordered)
                    db.execute(
                        """INSERT OR IGNORE INTO duplicate_reviews
                        (pair_key, left_id, right_id, event_ids_json, calendar_event_ids_json, reason, status, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)""",
                        (pair_key, ordered[0], ordered[1], _dump(event_ids), _dump(calendar_ids), issue.get("message", ""), now),
                    )
                    saved += 1
        return saved

    def pending_duplicate_reviews(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM duplicate_reviews WHERE status='pending' ORDER BY updated_at, left_id").fetchall()
        return self._review_rows(rows)

    @staticmethod
    def _review_rows(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
        return [
            {
                "pair_key": row["pair_key"],
                "left_id": row["left_id"],
                "right_id": row["right_id"],
                "event_ids": json.loads(row["event_ids_json"]),
                "calendar_event_ids": json.loads(row["calendar_event_ids_json"]),
                "reason": row["reason"],
                "status": row["status"],
            }
            for row in rows
        ]

    def list_reviews(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM duplicate_reviews ORDER BY updated_at DESC, left_id").fetchall()
        return self._review_rows(rows)

    def summary(self) -> dict[str, Any]:
        with self._connect() as db:
            counts = {
                "sources": db.execute("SELECT COUNT(*) FROM sources").fetchone()[0],
                "events": db.execute("SELECT COUNT(*) FROM events").fetchone()[0],
                "calendar_events": db.execute("SELECT COUNT(*) FROM calendar_events").fetchone()[0],
                "travel_estimates": db.execute("SELECT COUNT(*) FROM travel_estimates").fetchone()[0],
                "duplicate_reviews": db.execute("SELECT COUNT(*) FROM duplicate_reviews").fetchone()[0],
                "pending_duplicate_reviews": db.execute("SELECT COUNT(*) FROM duplicate_reviews WHERE status='pending'").fetchone()[0],
                "analysis_runs": db.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0],
            }
        return {"database": str(self.path.resolve()), "records": counts}

    def set_duplicate_review(self, pair_key: str, status: str) -> None:
        if status not in {"same", "different", "pending"}:
            raise ValueError("Review status must be same, different, or pending.")
        with self._connect() as db:
            cursor = db.execute("UPDATE duplicate_reviews SET status=?, updated_at=? WHERE pair_key=?", (status, datetime.now(timezone.utc).isoformat(), pair_key))
            if cursor.rowcount == 0:
                raise ValueError("Duplicate review not found. Run the duplicates command first.")

    def duplicate_review_statuses(self) -> dict[frozenset[str], str]:
        with self._connect() as db:
            rows = db.execute("SELECT left_id, right_id, status FROM duplicate_reviews WHERE status IN ('same','different')").fetchall()
        return {frozenset((row["left_id"], row["right_id"])): row["status"] for row in rows}

    def update_event_confirmation(self, event_id: str, user_status: str | None, event_status: str | None) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute("SELECT data_json FROM events WHERE event_id=?", (event_id,)).fetchone()
            if not row:
                raise ValueError(f"Event not found: {event_id}")
            event = json.loads(row[0])
            if user_status is not None:
                event["user_confirmation_status"] = user_status
            if event_status is not None:
                event["event_confirmation_status"] = event_status
            db.execute("UPDATE events SET data_json=? WHERE event_id=?", (_dump(event), event_id))
        return event

    def save_analysis(self, run_id: str, result: dict[str, Any]) -> None:
        with self._connect() as db:
            db.execute("INSERT INTO analysis_runs VALUES (?, ?, ?)", (run_id, datetime.now(timezone.utc).isoformat(), _dump(result)))

    def latest_analysis(self) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT result_json FROM analysis_runs ORDER BY created_at DESC LIMIT 1").fetchone()
        return json.loads(row[0]) if row else None

