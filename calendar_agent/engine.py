"""Deterministic calendar checks plus an optional Ollama duplicate check."""

from __future__ import annotations

import json
import re
import sys
from datetime import date, datetime
from typing import Any

from .ollama_client import OllamaClient, OllamaError


def _parse_datetime(value: str | None, label: str) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} is not a valid ISO datetime: {value}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include a timezone offset: {value}")
    return parsed


def _event_date(event: dict[str, Any], *, candidate: bool) -> str | None:
    if candidate:
        date_value = event.get("time", {}).get("date")
        if date_value:
            return str(date_value)
        start = event.get("time", {}).get("start")
    else:
        start = event.get("start")
    if not start:
        return None
    parsed = datetime.fromisoformat(start.replace("Z", "+00:00"))
    return parsed.date().isoformat()


def validate_input(data: dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise ValueError("Input JSON root must be an object.")
    required = {"schema_version", "request_id", "user", "sources", "event_candidates", "calendar_context"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"Missing top-level fields: {', '.join(sorted(missing))}")
    if not isinstance(data["user"], dict) or not isinstance(data["sources"], list) or not isinstance(data["calendar_context"], dict):
        raise ValueError("user/calendar_context must be objects and sources must be an array.")
    if data["schema_version"] != "1.0":
        raise ValueError("schema_version must be '1.0'.")
    if not isinstance(data["event_candidates"], list) or not isinstance(data["calendar_context"].get("events"), list):
        raise ValueError("event_candidates and calendar_context.events must be arrays.")

    all_ids: set[str] = set()
    source_ids = {source.get("source_id") for source in data["sources"]}
    for event in data["event_candidates"]:
        for key in ("event_id", "title", "type", "people", "organization", "time", "location", "content", "user_confirmation_status", "event_confirmation_status", "source_refs"):
            if key not in event:
                raise ValueError(f"Event candidate is missing '{key}'.")
        if event["event_id"] in all_ids:
            raise ValueError(f"Duplicate event_id: {event['event_id']}")
        all_ids.add(event["event_id"])
        for source_ref in event["source_refs"]:
            if source_ref not in source_ids:
                raise ValueError(f"Unknown source_ref '{source_ref}' in event '{event['event_id']}'.")
        time_data = event["time"]
        start = _parse_datetime(time_data.get("start"), f"{event['event_id']}.time.start")
        end = _parse_datetime(time_data.get("end"), f"{event['event_id']}.time.end")
        if (start is None) != (end is None):
            raise ValueError(f"Event '{event['event_id']}' must have both start and end, or neither.")
        if start and end and start >= end:
            raise ValueError(f"Event '{event['event_id']}' starts at or after its end.")
        date_value = time_data.get("date")
        if date_value:
            try:
                date.fromisoformat(date_value)
            except ValueError as exc:
                raise ValueError(f"Event '{event['event_id']}' has an invalid date: {date_value}") from exc
        if event["user_confirmation_status"] not in {"pending", "confirmed", "rejected"}:
            raise ValueError(f"Event '{event['event_id']}' has an invalid user_confirmation_status.")
        if event["event_confirmation_status"] not in {"confirmed", "tentative", "cancelled", "unknown"}:
            raise ValueError(f"Event '{event['event_id']}' has an invalid event_confirmation_status.")
        confidence = event.get("confidence")
        if confidence is not None and (not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1):
            raise ValueError(f"Event '{event['event_id']}' confidence must be between 0 and 1.")
        preparation_minutes = event.get("preparation_minutes")
        if preparation_minutes is not None and (not isinstance(preparation_minutes, int) or preparation_minutes < 0):
            raise ValueError(f"Event '{event['event_id']}' preparation_minutes must be a nonnegative integer.")

    for event in data["calendar_context"]["events"]:
        if event["event_id"] in all_ids:
            raise ValueError(f"event_id is duplicated across candidates and calendar: {event['event_id']}")
        all_ids.add(event["event_id"])
        start = _parse_datetime(event.get("start"), f"{event['event_id']}.start")
        end = _parse_datetime(event.get("end"), f"{event['event_id']}.end")
        if start is None or end is None or start >= end:
            raise ValueError(f"Calendar event '{event['event_id']}' has invalid start/end times.")


def _normalize(text: str | None) -> str:
    return re.sub(r"[^0-9a-z가-힣]+", "", (text or "").lower())


def _location_text(event: dict[str, Any]) -> str:
    location = event.get("location") or {}
    return _normalize(" ".join(str(location.get(key, "")) for key in ("name", "address", "place_id")))


def _title_tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[0-9a-z가-힣]+", text.lower()) if len(token) > 1}


def _candidate_pairs(data: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = data["event_candidates"]
    calendar_events = data["calendar_context"]["events"]
    entities: list[tuple[dict[str, Any], str]] = [(item, "candidate") for item in candidates]
    entities.extend((item, "calendar") for item in calendar_events)
    pairs: list[dict[str, Any]] = []

    for index, (left, left_kind) in enumerate(entities):
        if left_kind != "candidate":
            continue
        for right, right_kind in entities[index + 1 :]:
            if (left_kind == "candidate" and (left.get("event_confirmation_status") == "cancelled" or left.get("user_confirmation_status") == "rejected")) or (right_kind == "candidate" and (right.get("event_confirmation_status") == "cancelled" or right.get("user_confirmation_status") == "rejected")):
                continue
            left_date = _event_date(left, candidate=left_kind == "candidate")
            right_date = _event_date(right, candidate=right_kind == "candidate")
            left_tokens = _title_tokens(left.get("title", ""))
            right_tokens = _title_tokens(right.get("title", ""))
            token_overlap = bool(left_tokens & right_tokens)
            same_date = left_date is not None and left_date == right_date
            if not (same_date or token_overlap):
                continue
            pairs.append(
                {
                    "left_id": left["event_id"],
                    "right_id": right["event_id"],
                    "left": _for_model(left, left_kind),
                    "right": _for_model(right, right_kind),
                }
            )
    return pairs


def _for_model(event: dict[str, Any], kind: str) -> dict[str, Any]:
    return {
        "id": event["event_id"],
        "record_type": kind,
        "title": event.get("title"),
        "type": event.get("type"),
        "people": event.get("people", event.get("attendees", [])),
        "organization": event.get("organization"),
        "date": _event_date(event, candidate=kind == "candidate"),
        "start": event.get("time", {}).get("start") if kind == "candidate" else event.get("start"),
        "end": event.get("time", {}).get("end") if kind == "candidate" else event.get("end"),
        "location": event.get("location"),
        "content": event.get("content"),
    }


def _heuristic_duplicate_pairs(pairs: list[dict[str, Any]]) -> list[dict[str, str]]:
    matches = []
    for pair in pairs:
        left, right = pair["left"], pair["right"]
        same_title = _normalize(left.get("title")) == _normalize(right.get("title"))
        same_date = left.get("date") and left.get("date") == right.get("date")
        same_location = _location_text(left) and _location_text(left) == _location_text(right)
        same_start = left.get("start") and left.get("start") == right.get("start")
        if same_date and same_location and (same_start or same_title):
            matches.append({"left_id": pair["left_id"], "right_id": pair["right_id"], "relation": "possible_duplicate", "reason": "날짜와 장소가 같고 일정명 또는 시작 시각도 일치함(로컬 판정)."})
    return matches


def _event_interval(event: dict[str, Any], candidate: bool) -> tuple[datetime | None, datetime | None]:
    if candidate:
        time_data = event["time"]
        return (_parse_datetime(time_data.get("start"), "candidate.start"), _parse_datetime(time_data.get("end"), "candidate.end"))
    return (_parse_datetime(event.get("start"), "calendar.start"), _parse_datetime(event.get("end"), "calendar.end"))


def _event_index(data: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], set[str]]:
    candidates = {item["event_id"]: item for item in data["event_candidates"]}
    all_events = dict(candidates)
    all_events.update({item["event_id"]: item for item in data["calendar_context"]["events"]})
    return all_events, set(candidates)


def _make_question(question_id: str, kind: str, event_ids: list[str], prompt: str, choices: list[str] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"question_id": question_id, "type": kind, "event_ids": event_ids, "prompt": prompt, "blocking": True}
    if choices:
        result["choices"] = choices
    return result


def run_workflow(
    data: dict[str, Any],
    *,
    model: str = "qwen3.5:4b",
    ollama_base_url: str = "http://127.0.0.1:11434",
    use_llm: bool = True,
    duplicate_review_statuses: dict[frozenset[str], str] | None = None,
) -> dict[str, Any]:
    validate_input(data)
    candidates = data["event_candidates"]
    calendar_events = data["calendar_context"]["events"]
    all_events, candidate_ids = _event_index(data)
    warnings: list[str] = []
    llm_status = "disabled" if not use_llm else "ok"

    pairs = _candidate_pairs(data)
    if use_llm and pairs:
        try:
            classifications = OllamaClient(model=model, base_url=ollama_base_url).classify_duplicate_pairs(pairs)
        except OllamaError as exc:
            llm_status = "unavailable"
            warnings.append(str(exc))
            print(f"Warning: {exc}; using deterministic duplicate fallback.", file=sys.stderr)
            classifications = _heuristic_duplicate_pairs(pairs)
    elif use_llm:
        llm_status = "skipped_no_pairs"
        classifications = []
    else:
        classifications = _heuristic_duplicate_pairs(pairs)

    valid_pairs = {frozenset((pair["left_id"], pair["right_id"])): pair for pair in pairs}
    review_statuses = duplicate_review_statuses or {}
    classifications_by_pair = {
        frozenset((match.get("left_id"), match.get("right_id"))): match
        for match in classifications
        if match.get("left_id") and match.get("right_id")
    }
    effective_classifications: list[dict[str, str]] = []
    for pair_key, pair_data in valid_pairs.items():
        review_status = review_statuses.get(pair_key)
        if review_status == "different":
            continue
        if review_status == "same":
            effective_classifications.append({"left_id": pair_data["left_id"], "right_id": pair_data["right_id"], "relation": "possible_duplicate", "reason": "사용자가 동일 일정으로 확인함."})
            continue
        match = classifications_by_pair.get(pair_key)
        if match and match.get("relation") in {"same_event", "possible_duplicate", "uncertain"}:
            effective_classifications.append(match)

    duplicates_by_candidate: dict[str, list[dict[str, str]]] = {}
    duplicate_relation_pairs: set[frozenset[str]] = set()
    for match in effective_classifications:
        left_id, right_id = match.get("left_id"), match.get("right_id")
        pair_key = frozenset((left_id, right_id))
        duplicate_relation_pairs.add(pair_key)
        candidate_id = left_id if left_id in candidate_ids else right_id if right_id in candidate_ids else None
        other_id = right_id if candidate_id == left_id else left_id
        if candidate_id:
            duplicates_by_candidate.setdefault(candidate_id, []).append({"other_id": other_id, "relation": match["relation"], "reason": str(match.get("reason", "유사한 일정이 발견됨.")), "review_status": review_statuses.get(pair_key, "pending")})

    issues: list[dict[str, Any]] = []
    questions: list[dict[str, Any]] = []
    questioned_event_ids: set[str] = set()
    blocked: set[str] = set()
    issue_number = 1
    question_number = 1

    def add_issue(issue_type: str, severity: str, event_ids: list[str], calendar_ids: list[str], message: str) -> None:
        nonlocal issue_number
        issues.append({"issue_id": f"iss-{issue_number:03d}", "type": issue_type, "severity": severity, "event_ids": sorted(set(event_ids)), "calendar_event_ids": sorted(set(calendar_ids)), "message": message})
        issue_number += 1
        if severity == "blocking":
            blocked.update(event_ids)

    def add_question(kind: str, event_ids: list[str], prompt: str, choices: list[str] | None = None) -> None:
        nonlocal question_number
        unique_event_ids = sorted(set(event_ids))
        questions.append(_make_question(f"q-{question_number:03d}", kind, unique_event_ids, prompt, choices))
        questioned_event_ids.update(unique_event_ids)
        question_number += 1

    # Duplicate classification is always a request for confirmation, never an automatic merge.
    for candidate_id, duplicates in duplicates_by_candidate.items():
        for duplicate in duplicates:
            other_id = duplicate["other_id"]
            other_calendar = other_id not in candidate_ids
            calendar_ids = [other_id] if other_calendar else []
            add_issue("possible_duplicate", "blocking", [candidate_id] + ([] if other_calendar else [other_id]), calendar_ids, duplicate["reason"])
            if duplicate.get("review_status") != "same":
                add_question("confirm_duplicate", [candidate_id] + ([] if other_calendar else [other_id]), f"'{all_events[candidate_id]['title']}'이(가) '{all_events[other_id]['title']}' 일정과 같은 건가요?", ["같은 일정이라 새로 등록하지 않음", "별도 일정으로 등록"])

    # Compare all scheduled intervals, including candidate-candidate and candidate-calendar pairs.
    scheduled: list[tuple[dict[str, Any], str, datetime, datetime]] = []
    for candidate in candidates:
        start, end = _event_interval(candidate, True)
        if start and end and candidate["event_confirmation_status"] != "cancelled" and candidate["user_confirmation_status"] != "rejected":
            scheduled.append((candidate, "candidate", start, end))
    for event in calendar_events:
        start, end = _event_interval(event, False)
        if start and end:
            scheduled.append((event, "calendar", start, end))
    scheduled.sort(key=lambda item: item[2])

    pair_seen: set[tuple[str, str]] = set()
    for index, left in enumerate(scheduled):
        for right in scheduled[index + 1 :]:
            left_event, left_kind, left_start, left_end = left
            right_event, right_kind, right_start, right_end = right
            if right_start >= left_end:
                break
            if left_kind == right_kind == "calendar":
                continue
            pair = tuple(sorted((left_event["event_id"], right_event["event_id"])))
            if pair in pair_seen:
                continue
            pair_seen.add(pair)
            if frozenset(pair) in duplicate_relation_pairs:
                continue
            event_ids = [event_id for event_id in pair if event_id in candidate_ids]
            calendar_ids = [event_id for event_id in pair if event_id not in candidate_ids]
            if not event_ids:
                continue
            add_issue("time_conflict", "blocking", event_ids, calendar_ids, f"'{left_event['title']}'과(와) '{right_event['title']}'의 시간이 겹칩니다.")
            add_question("resolve_conflict", event_ids, f"'{left_event['title']}'과(와) '{right_event['title']}'의 시간이 겹칩니다. 어떤 일정을 조정할까요?", ["첫 번째 일정 조정", "두 번째 일정 조정", "둘 다 유지하고 직접 조정"])

    # For each candidate, check its closest non-overlapping predecessor and successor.
    travel_lookup = {
        (item["from_event_id"], item["to_event_id"]): item
        for item in data["calendar_context"].get("travel_estimates", [])
    }
    buffer_minutes = int(data["user"].get("minimum_buffer_minutes", 10))
    travel_pairs: set[tuple[str, str]] = set()
    for candidate in candidates:
        candidate_id = candidate["event_id"]
        candidate_interval = next((item for item in scheduled if item[0]["event_id"] == candidate_id), None)
        if candidate_interval is None:
            continue
        _, _, candidate_start, candidate_end = candidate_interval
        predecessors = [item for item in scheduled if item[0]["event_id"] != candidate_id and item[3] <= candidate_start]
        successors = [item for item in scheduled if item[0]["event_id"] != candidate_id and item[2] >= candidate_end]
        neighbors = []
        if predecessors:
            closest_predecessor = max(predecessors, key=lambda item: item[3])
            if (candidate_start - closest_predecessor[3]).total_seconds() <= 6 * 60 * 60:
                neighbors.append((closest_predecessor, candidate_interval))
        if successors:
            closest_successor = min(successors, key=lambda item: item[2])
            if (closest_successor[2] - candidate_end).total_seconds() <= 6 * 60 * 60:
                neighbors.append((candidate_interval, closest_successor))
        for left, right in neighbors:
            prev_event, prev_kind, _, prev_end = left
            next_event, next_kind, next_start, _ = right
            prev_id, next_id = prev_event["event_id"], next_event["event_id"]
            pair_key = (prev_id, next_id)
            if pair_key in travel_pairs:
                continue
            travel_pairs.add(pair_key)
            if not _location_text(prev_event) or not _location_text(next_event) or _location_text(prev_event) == _location_text(next_event):
                continue
            if prev_kind == next_kind == "calendar":
                continue
            event_ids = [event_id for event_id in (prev_id, next_id) if event_id in candidate_ids]
            calendar_ids = [event_id for event_id in (prev_id, next_id) if event_id not in candidate_ids]
            if not event_ids:
                continue
            estimate = travel_lookup.get((prev_id, next_id))
            gap_minutes = int((next_start - prev_end).total_seconds() // 60)
            if estimate is None:
                add_issue("travel_time", "blocking", event_ids, calendar_ids, f"'{prev_event['title']}'에서 '{next_event['title']}'로 이동하는 데 필요한 시간을 알 수 없습니다.")
                add_question("choose_travel_mode", event_ids, "두 일정 사이 이동수단 또는 예상 이동시간을 확인해 주세요.", ["대중교통", "자동차", "도보", "이동시간 직접 입력"])
            elif gap_minutes < int(estimate["minutes"]) + buffer_minutes:
                add_issue("travel_time", "blocking", event_ids, calendar_ids, f"'{prev_event['title']}' 종료 후 '{next_event['title']}'까지 이동에 {estimate['minutes']}분이 필요하지만, 일정 간격은 {gap_minutes}분입니다(버퍼 {buffer_minutes}분 포함 필요).")
                add_question("resolve_conflict", event_ids, f"이동시간을 포함하면 '{next_event['title']}'에 늦을 수 있습니다. 일정을 조정할까요?", ["다음 일정 늦추기", "다른 날짜로 옮기기", "그대로 두기"])

    # Missing time and unconfirmed status are explicit blockers.
    for candidate in candidates:
        event_id = candidate["event_id"]
        if candidate["event_confirmation_status"] == "cancelled" or candidate["user_confirmation_status"] == "rejected":
            continue
        start, end = _event_interval(candidate, True)
        if not start or not end:
            add_issue("missing_information", "blocking", [event_id], [], "시작·종료 시각이 없어 충돌 및 이동시간을 확인할 수 없습니다.")
            add_question("choose_time", [event_id], f"'{candidate['title']}'의 정확한 시작·종료 시각을 알려주세요.")
        if (candidate["user_confirmation_status"] != "confirmed" or candidate["event_confirmation_status"] != "confirmed") and event_id not in questioned_event_ids:
            add_question("confirm_details", [event_id], f"'{candidate['title']}' 일정 정보를 확인하고 캘린더 등록을 승인할까요?", ["확인 및 등록 승인", "수정 필요", "등록하지 않음"])
            blocked.add(event_id)

    # Assemble one result per candidate. A candidate with any blocker cannot be written.
    proposals: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    for candidate in candidates:
        event_id = candidate["event_id"]
        start, end = _event_interval(candidate, True)
        duplicates = duplicates_by_candidate.get(event_id, [])
        cancelled = candidate["event_confirmation_status"] == "cancelled" or candidate["user_confirmation_status"] == "rejected"
        if cancelled:
            status, reason = "unscheduled", "취소되었거나 사용자가 등록을 거절함"
        elif duplicates:
            status, reason = "duplicate_suspected", duplicates[0]["reason"]
        elif not start or not end:
            status, reason = "unscheduled", "시작·종료 시각이 없어 일정 시간을 확정할 수 없음"
        elif event_id in blocked:
            status, reason = "needs_confirmation", "확인 질문 또는 blocking 이슈가 남아 있음"
        else:
            status, reason = "scheduled", "필수 확인이 끝났고 충돌·이동 문제 없음"

        proposal: dict[str, Any] = {"event_id": event_id, "title": candidate["title"], "status": status, "reason": reason, "source_refs": candidate["source_refs"]}
        if start and end:
            proposal["start"], proposal["end"] = start.isoformat(), end.isoformat()
        elif candidate["time"].get("date"):
            proposal["date"] = candidate["time"]["date"]
        if candidate.get("location"):
            proposal["location"] = candidate["location"]
        proposals.append(proposal)

        create = status == "scheduled" and candidate["user_confirmation_status"] == "confirmed" and candidate["event_confirmation_status"] == "confirmed"
        actions.append({
            "action_id": f"act-{len(actions) + 1:03d}",
            "type": "create" if create else "skip",
            "event_id": event_id,
            "approval_required": not create,
            "idempotency_key": f"{data['request_id']}:{event_id}:{'create' if create else 'pending'}",
        })

    if questions or any(issue["severity"] == "blocking" for issue in issues):
        result_status = "needs_user_input"
    elif any(action["type"] == "create" for action in actions):
        result_status = "ready"
    else:
        result_status = "no_action"

    return {
        "schema_version": "1.0",
        "request_id": data["request_id"],
        "result_status": result_status,
        "proposed_events": proposals,
        "issues": issues,
        "questions": questions,
        "calendar_actions": actions,
        "diagnostics": {"llm_status": llm_status, "llm_model": model if use_llm else "", "warnings": warnings},
    }

