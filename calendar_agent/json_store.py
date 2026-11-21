"""JSON-file persistence for imported and reviewed calendar events."""

from __future__ import annotations

import json
import hashlib
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .engine import validate_input


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class JsonStore:
    """Keep application data in a JSON file separate from source input JSON."""

    def __init__(self, path: str | Path = "data/registered_events.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.reset()
        self._read()

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {
            "store_version": 1,
            "events": [],
            "sources": [],
            "user": {"timezone": "Asia/Seoul", "minimum_buffer_minutes": 10},
            "calendar_context": {"range_start": None, "range_end": None, "events": [], "travel_estimates": []},
            "duplicate_decisions": {},
            "removed_duplicate_ids": {},
            "deleted_event_ids": {},
            "import_fingerprints": {},
            "reviewed_fingerprints": {},
            "removed_events": [],
            "review_issues": [],
        }

    def _read(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"저장 JSON을 읽을 수 없습니다: {self.path} ({exc})") from exc
        if not isinstance(data, dict) or not isinstance(data.get("events"), list):
            raise ValueError(f"저장 JSON 구조가 올바르지 않습니다: {self.path}")
        return data

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)

    @staticmethod
    def _refresh_source_statuses(data: dict[str, Any]) -> None:
        all_events = data.get("events", []) + data.get("removed_events", [])
        for source in data.get("sources", []):
            linked = [event for event in all_events if source.get("source_id") in event.get("source_refs", [])]
            if not linked:
                source["inbox_status"] = "detected"
            elif any(event.get("workflow_status") not in {"confirmed", "ignored", "duplicate_removed", "deleted"} for event in linked):
                source["inbox_status"] = "review_pending"
            else:
                source["inbox_status"] = "processed"

    def reset(self) -> None:
        self._write(self._empty())

    def events(self) -> list[dict[str, Any]]:
        return deepcopy(self._read()["events"])

    def sources(self) -> list[dict[str, Any]]:
        return deepcopy(self._read().get("sources", []))

    def travel_estimates(self) -> list[dict[str, Any]]:
        return deepcopy(self._read().get("calendar_context", {}).get("travel_estimates", []))

    def review_issues(self) -> list[dict[str, Any]]:
        return deepcopy(self._read().get("review_issues", []))

    def import_document(self, document: dict[str, Any]) -> dict[str, Any]:
        validate_input(document)
        data = self._read()
        now = _now()
        existing = {event["event_id"]: event for event in data["events"]}
        source_map = {source["source_id"]: source for source in document["sources"]}
        import_fingerprints = data.setdefault("import_fingerprints", {})
        reviewed_fingerprints = data.setdefault("reviewed_fingerprints", {})
        added = updated = existing_count = 0
        review_event_ids: list[str] = []
        for source_event in document["event_candidates"]:
            if (source_event["event_id"] in data.get("removed_duplicate_ids", {})
                    or source_event["event_id"] in data.get("deleted_event_ids", {})):
                existing_count += 1
                continue
            event_id = source_event["event_id"]
            source_payload = [source_map[source_id] for source_id in source_event.get("source_refs", []) if source_id in source_map]
            fingerprint_payload = json.dumps(
                {"event": source_event, "sources": source_payload},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            fingerprint = hashlib.sha256(fingerprint_payload).hexdigest()
            old = existing.get(source_event["event_id"])
            if old:
                status = old.get("workflow_status")
                previous_fingerprint = reviewed_fingerprints.get(event_id)
                if status == "ignored":
                    existing_count += 1
                    continue
                if status in {"ready", "confirmed"} and previous_fingerprint is None:
                    # One-time migration for schedules reviewed before fingerprint tracking.
                    import_fingerprints[event_id] = fingerprint
                    reviewed_fingerprints[event_id] = fingerprint
                    old["workflow_status"] = "confirmed"
                    old["user_confirmation_status"] = "confirmed"
                    old["event_confirmation_status"] = "confirmed"
                    existing_count += 1
                    continue
                if status in {"duplicate_review", "ready"} and (
                    import_fingerprints.get(event_id) == fingerprint or previous_fingerprint == fingerprint
                ):
                    old["workflow_status"] = "confirmed"
                    old["user_confirmation_status"] = "confirmed"
                    old["event_confirmation_status"] = "confirmed"
                    import_fingerprints[event_id] = fingerprint
                    reviewed_fingerprints[event_id] = fingerprint
                    existing_count += 1
                    continue
                if status != "conflict_review" and previous_fingerprint == fingerprint:
                    import_fingerprints[event_id] = fingerprint
                    existing_count += 1
                    continue

                if import_fingerprints.get(event_id) != fingerprint:
                    event = deepcopy(source_event)
                    event["registered_at"] = old.get("registered_at", now)
                    event["workflow_status"] = "conflict_review"
                    index = next(i for i, item in enumerate(data["events"]) if item["event_id"] == event_id)
                    data["events"][index] = event
                    import_fingerprints[event_id] = fingerprint
                    data["duplicate_decisions"] = {
                        key: value for key, value in data.get("duplicate_decisions", {}).items()
                        if event_id not in key.split("|", 1)
                    }
                    updated += 1
                review_event_ids.append(event_id)
            else:
                event = deepcopy(source_event)
                event["registered_at"] = now
                event["workflow_status"] = "conflict_review"
                data["events"].append(event)
                import_fingerprints[event["event_id"]] = fingerprint
                review_event_ids.append(event["event_id"])
                added += 1

        sources = {item["source_id"]: item for item in data.get("sources", [])}
        for item in document["sources"]:
            incoming = deepcopy(item)
            previous = sources.get(item["source_id"])
            incoming["inbox_status"] = previous.get("inbox_status", "review_pending") if previous else "review_pending"
            sources[item["source_id"]] = incoming
        data["sources"] = list(sources.values())
        data["user"] = deepcopy(document["user"])

        new_context = document["calendar_context"]
        data["calendar_context"] = {
            "range_start": new_context.get("range_start"),
            "range_end": new_context.get("range_end"),
            # Calendar entries from outside this app are no longer imported or retained.
            "events": [],
            "travel_estimates": [],
        }
        self._refresh_source_statuses(data)
        self._write(data)
        return {
            "added": added,
            "updated": updated,
            "existing": existing_count,
            "total": len(data["events"]),
            "review_event_ids": review_event_ids,
        }

    def mark_conflicts_clear(self, event_ids: list[str]) -> int:
        """Confirm schedules immediately after their blocking conflict checks pass."""
        data = self._read()
        target_ids = set(event_ids)
        imported = data.setdefault("import_fingerprints", {})
        reviewed = data.setdefault("reviewed_fingerprints", {})
        confirmed_count = 0
        for event in data["events"]:
            if event.get("event_id") in target_ids and event.get("workflow_status") == "conflict_review":
                event["workflow_status"] = "confirmed"
                event["user_confirmation_status"] = "confirmed"
                event["event_confirmation_status"] = "confirmed"
                event_id = event["event_id"]
                if event_id in imported:
                    reviewed[event_id] = imported[event_id]
                confirmed_count += 1
        self._refresh_source_statuses(data)
        self._write(data)
        return confirmed_count

    def build_input(self, request_id: str) -> dict[str, Any]:
        data = self._read()
        active_events = [event for event in data["events"] if event.get("workflow_status") != "ignored"]
        if not active_events:
            raise ValueError("저장된 일정이 없습니다. 먼저 일정 데이터를 가져오세요.")
        context = deepcopy(data.get("calendar_context", {}))
        if not context.get("range_start") or not context.get("range_end"):
            raise ValueError("입력 JSON에 캘린더 조회 기간이 없습니다.")
        events = []
        for stored in active_events:
            event = deepcopy(stored)
            event.pop("registered_at", None)
            event.pop("workflow_status", None)
            event.pop("duplicate_of", None)
            event.pop("google_calendar_event_id", None)
            event.pop("google_calendar_html_link", None)
            event.pop("google_calendar_synced_at", None)
            event.pop("google_sync_status", None)
            events.append(event)
        result = {
            "schema_version": "1.0",
            "request_id": request_id,
            "user": data["user"],
            "sources": [
                {key: value for key, value in source.items() if key != "inbox_status"}
                for source in data.get("sources", [])
            ],
            "event_candidates": events,
            "calendar_context": {
                "range_start": context["range_start"],
                "range_end": context["range_end"],
                "events": [],
                "travel_estimates": [
                    estimate for estimate in context.get("travel_estimates", [])
                    if estimate.get("from_event_id") in {event["event_id"] for event in events}
                    and estimate.get("to_event_id") in {event["event_id"] for event in events}
                ],
            },
        }
        validate_input(result)
        return result

    def update_event(self, index: int, updates: dict[str, Any]) -> dict[str, Any]:
        data = self._read()
        if index < 1 or index > len(data["events"]):
            raise ValueError(f"일정 번호가 올바르지 않습니다: {index}")
        event = data["events"][index - 1]
        for key, value in updates.items():
            if key == "time":
                event["time"].update(value)
            elif key == "location":
                event["location"] = value
            else:
                event[key] = value
        event["workflow_status"] = "conflict_review"
        event_id = event["event_id"]
        data["duplicate_decisions"] = {
            key: value for key, value in data.get("duplicate_decisions", {}).items()
            if event_id not in key.split("|", 1)
        }
        data["review_issues"] = [
            issue for issue in data.get("review_issues", [])
            if event_id not in issue.get("event_ids", [])
        ]
        context = data.get("calendar_context", {})
        candidate = {
            "schema_version": "1.0",
            "request_id": "validation",
            "user": data["user"],
            "sources": [
                {key: value for key, value in source.items() if key != "inbox_status"}
                for source in data.get("sources", [])
            ],
            "event_candidates": [
                {key: value for key, value in item.items() if key not in {
                    "registered_at", "workflow_status", "duplicate_of",
                    "google_calendar_event_id", "google_calendar_html_link",
                    "google_calendar_synced_at", "google_sync_status",
                }}
                for item in data["events"]
            ],
            "calendar_context": context,
        }
        validate_input(candidate)
        self._refresh_source_statuses(data)
        self._write(data)
        return deepcopy(event)

    def ignore_event(self, index: int) -> dict[str, Any]:
        data = self._read()
        if index < 1 or index > len(data["events"]):
            raise ValueError(f"일정 번호가 올바르지 않습니다: {index}")
        event = data["events"][index - 1]
        event["workflow_status"] = "ignored"
        event["user_confirmation_status"] = "rejected"
        event_id = event["event_id"]
        data["review_issues"] = [
            issue for issue in data.get("review_issues", [])
            if event_id not in issue.get("event_ids", [])
        ]
        self._refresh_source_statuses(data)
        self._write(data)
        return deepcopy(event)

    def delete_event(self, index: int) -> dict[str, Any]:
        """Remove a candidate from active storage while retaining a source-status tombstone."""
        data = self._read()
        if index < 1 or index > len(data["events"]):
            raise ValueError(f"일정 번호가 올바르지 않습니다: {index}")
        event = data["events"].pop(index - 1)
        event["workflow_status"] = "deleted"
        data.setdefault("removed_events", []).append(event)
        data.setdefault("deleted_event_ids", {})[event["event_id"]] = True
        data["review_issues"] = [
            issue for issue in data.get("review_issues", [])
            if event["event_id"] not in issue.get("event_ids", [])
        ]
        self._refresh_source_statuses(data)
        self._write(data)
        return deepcopy(event)

    def save_review_issues(self, issues: list[dict[str, Any]]) -> None:
        data = self._read()
        data["review_issues"] = deepcopy(issues)
        self._write(data)

    def remove_duplicate(self, index: int, other_id: str) -> dict[str, Any]:
        data = self._read()
        if index < 1 or index > len(data["events"]):
            raise ValueError(f"일정 번호가 올바르지 않습니다: {index}")
        removed = data["events"].pop(index - 1)
        removed["workflow_status"] = "duplicate_removed"
        data.setdefault("removed_events", []).append(removed)
        data.setdefault("duplicate_decisions", {})["|".join(sorted((removed["event_id"], other_id)))] = "same"
        data.setdefault("removed_duplicate_ids", {})[removed["event_id"]] = other_id
        self._refresh_source_statuses(data)
        self._write(data)
        return removed

    def set_duplicate_decision(self, left_id: str, right_id: str, decision: str) -> None:
        data = self._read()
        data.setdefault("duplicate_decisions", {})["|".join(sorted((left_id, right_id)))] = decision
        self._write(data)

    def duplicate_review_statuses(self) -> dict[frozenset[str], str]:
        data = self._read()
        result = {}
        for key, status in data.get("duplicate_decisions", {}).items():
            ids = key.split("|", 1)
            if len(ids) == 2:
                result[frozenset(ids)] = status
        return result

    def add_travel_estimate(self, from_id: str, to_id: str, minutes: int, mode: str) -> None:
        if minutes < 0:
            raise ValueError("이동시간은 0분 이상이어야 합니다.")
        data = self._read()
        estimates = data["calendar_context"].setdefault("travel_estimates", [])
        estimate = {"from_event_id": from_id, "to_event_id": to_id, "minutes": minutes, "mode": mode}
        for i, item in enumerate(estimates):
            if item["from_event_id"] == from_id and item["to_event_id"] == to_id:
                estimates[i] = estimate
                break
        else:
            estimates.append(estimate)
        self._write(data)


