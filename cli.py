"""Interactive menu for importing, reviewing, confirming, editing, and listing events."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any

from calendar_agent.engine import run_workflow
from calendar_agent.json_store import JsonStore
from calendar_agent.ollama_client import OllamaClient, OllamaError


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="일정 에이전트 대화형 CLI")
    parser.add_argument("--store", default="data/registered_events.json", help="우리 측 일정 저장 JSON 경로")
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "qwen3.5:4b"))
    parser.add_argument("--ollama-url", default=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    return parser.parse_args()


def _run(store: JsonStore, *, use_llm: bool, model: str, ollama_url: str) -> dict[str, Any]:
    return run_workflow(
        store.build_input(f"cli-{uuid.uuid4()}"),
        model=model,
        ollama_base_url=ollama_url,
        use_llm=use_llm,
        duplicate_review_statuses=store.duplicate_review_statuses(),
    )


def _index_by_id(store: JsonStore) -> dict[str, int]:
    return {event["event_id"]: i for i, event in enumerate(store.events(), 1)}


def _show_events(store: JsonStore) -> None:
    events = store.events()
    if not events:
        print("등록된 일정이 없습니다.")
        return
    print("\n=== 우리 저장소의 일정 ===")
    for index, event in enumerate(events, 1):
        time_data = event["time"]
        time_text = f"{time_data.get('start') or time_data.get('date') or '날짜 미정'} ~ {time_data.get('end') or ''}"
        confidence = event.get("confidence")
        confidence_text = f" | AI 신뢰도: {confidence:.0%}" if isinstance(confidence, (int, float)) else ""
        print(f"[{index}] {event['title']} | {time_text} | 상태: {event.get('workflow_status', 'conflict_review')}{confidence_text} | 등록: {event.get('registered_at', '-')}")
        print(f"    ID: {event['event_id']} | 장소: {(event.get('location') or {}).get('name', '-')}")


def _calendar_timeline(store: JsonStore) -> None:
    events = store.events()
    sources = {source["source_id"]: source for source in store.sources()}
    issues = store.review_issues()
    total = len([event for event in events if event.get("workflow_status") != "ignored"])
    pending = sum(event.get("workflow_status") in {"conflict_review", "duplicate_review", "ready"} for event in events)
    conflict_ids = {event_id for issue in issues if issue.get("severity") == "blocking" for event_id in issue.get("event_ids", [])}
    confirmed = sum(event.get("workflow_status") == "confirmed" for event in events)
    conflict_free = sum(event.get("workflow_status") in {"duplicate_review", "ready", "confirmed"} for event in events)
    print(f"\n전체 {total} | 검토 대기 {pending} | 충돌 관련 일정 {len(conflict_ids)} | 충돌 없음 {conflict_free} | 확정 {confirmed}")
    day = input("조회 날짜 YYYY-MM-DD (Enter=오늘, all=전체): ").strip()
    if not day:
        day = date.today().isoformat()
    elif day.lower() == "all":
        day = ""
    if day:
        try:
            date.fromisoformat(day)
        except ValueError as exc:
            raise ValueError("날짜를 YYYY-MM-DD 형식으로 입력하세요.") from exc

    rows: list[tuple[str, str, str, str, str]] = []
    for event in events:
        if event.get("workflow_status") != "confirmed":
            continue
        time_data = event.get("time", {})
        start = time_data.get("start") or ""
        end = time_data.get("end") or ""
        event_day = start[:10] or time_data.get("date") or ""
        if day and event_day != day:
            continue
        platform = ", ".join(sorted({sources.get(ref, {}).get("platform", "") for ref in event.get("source_refs", []) if sources.get(ref, {}).get("platform")})) or "직접 입력"
        rows.append((event_day, start, end, event["title"], f"{platform} | 내부 등록 | {event['event_id']}"))

    if not rows:
        print("표시할 일정이 없습니다.")
        return
    print("\n=== 메인 일정 타임라인 ===")
    for event_day, start, end, title, origin in sorted(rows, key=lambda row: (row[0], row[1], row[3])):
        start_time = start[11:16] if len(start) >= 16 else "시간 미정"
        end_time = end[11:16] if len(end) >= 16 else ""
        time_text = f"{start_time}–{end_time}" if end_time else start_time
        when = f"{event_day} {time_text}".strip() if event_day else time_text
        print(f"{when}  {title}  [{origin}]")


def _summarize_day(store: JsonStore, model: str, ollama_url: str) -> None:
    events_by_day: dict[str, list[dict[str, Any]]] = {}
    for event in store.events():
        if event.get("workflow_status") != "confirmed":
            continue
        time_data = event.get("time", {})
        start = time_data.get("start") or ""
        event_day = start[:10] or time_data.get("date") or ""
        if not event_day:
            continue
        events_by_day.setdefault(event_day, []).append({
            "title": event.get("title"),
            "type": event.get("type"),
            "start": start or None,
            "end": time_data.get("end"),
            "date": time_data.get("date"),
            "timezone": time_data.get("timezone"),
            "location": event.get("location"),
            "people": event.get("people", []),
            "content": event.get("content"),
            "preparation_minutes": event.get("preparation_minutes", 0),
        })
    if not events_by_day:
        print("날짜 요약을 할 내부 확정 일정이 없습니다.")
        return

    days = sorted(events_by_day)
    print("\n요약할 날짜를 선택하세요:")
    for index, event_day in enumerate(days, 1):
        titles = ", ".join(event.get("title") or "제목 없음" for event in events_by_day[event_day])
        print(f"[{index}] {event_day} | {len(events_by_day[event_day])}건 | {titles}")
    raw_choice = input("날짜 번호 (q=취소): ").strip().lower()
    if raw_choice == "q":
        print("날짜 요약을 취소했습니다.")
        return
    if not raw_choice.isdigit() or not 1 <= int(raw_choice) <= len(days):
        raise ValueError("목록에 표시된 날짜 번호를 선택하세요.")
    day = days[int(raw_choice) - 1]
    matched = events_by_day[day]
    matched.sort(key=lambda event: (event.get("start") or "", event.get("title") or ""))

    print(f"\n[{day} 일정 요약 중...]")
    try:
        summary = OllamaClient(model=model, base_url=ollama_url).summarize_schedule(day, matched)
    except OllamaError as exc:
        print(f"AI 요약을 만들지 못했습니다: {exc}")
        print("Ollama가 실행 중인지, 선택한 모델이 설치되어 있는지 확인하세요.")
        return
    print(f"\n=== {day} AI 일정 요약 ===\n{summary}")


def _analyze_event_flows(store: JsonStore, model: str, ollama_url: str) -> None:
    events = [event for event in store.events() if event.get("workflow_status") == "confirmed"]
    if len(events) < 2:
        print("연계 분석을 하려면 내부 캘린더에 확정 일정이 2개 이상 있어야 합니다.")
        return

    sources = {source["source_id"]: source for source in store.sources()}
    payload = []
    for event in events:
        linked_sources = [sources[ref] for ref in event.get("source_refs", []) if ref in sources]
        payload.append({
            "event_id": event["event_id"],
            "title": event.get("title"),
            "type": event.get("type"),
            "time": event.get("time"),
            "organization": event.get("organization"),
            "people": event.get("people", []),
            "location": event.get("location"),
            "content": event.get("content"),
            "source_notes": [
                {"subject": source.get("subject"), "raw_text": source.get("raw_text")}
                for source in linked_sources
            ],
        })

    print("\n[연계 일정·프로젝트 흐름 분석 중...]")
    try:
        flows = OllamaClient(model=model, base_url=ollama_url).analyze_event_flows(payload)
    except OllamaError as exc:
        print(f"AI 연계 분석을 만들지 못했습니다: {exc}")
        print("Ollama가 실행 중인지, 선택한 모델이 설치되어 있는지 확인하세요.")
        return

    event_map = {event["event_id"]: event for event in events}
    linked_ids: set[str] = set()
    displayed = 0
    print("\n=== AI가 찾은 일정 흐름 ===")
    for flow in flows:
        raw_ids = flow.get("event_ids", [])
        if not isinstance(raw_ids, list):
            continue
        flow_ids = list(dict.fromkeys(
            event_id for event_id in raw_ids
            if isinstance(event_id, str) and event_id in event_map
        ))
        if len(flow_ids) < 2:
            continue
        displayed += 1
        linked_ids.update(flow_ids)
        confidence = flow.get("confidence", "low")
        title = flow.get("title") or "연계 일정"
        print(f"\n{displayed}. {title} | 신뢰도: {confidence}")
        for event_id in flow_ids:
            event = event_map[event_id]
            time_data = event.get("time", {})
            start = time_data.get("start") or ""
            end = time_data.get("end") or ""
            day = start[:10] or time_data.get("date") or "날짜 미정"
            start_text = start[11:16] if len(start) >= 16 else "시간 미정"
            end_text = end[11:16] if len(end) >= 16 else ""
            time_text = f"{start_text}–{end_text}" if end_text else start_text
            print(f"  {day} {time_text} | [{event_id}] {event['title']}")
        print(f"  분석 근거: {flow.get('summary') or '제공된 일정 정보에서 연계 근거를 찾았습니다.'}")

    if not displayed:
        print("입력된 정보만으로 연결 관계가 분명한 일정 흐름을 찾지 못했습니다.")
    else:
        unrelated = [event for event in events if event["event_id"] not in linked_ids]
        if unrelated:
            print("\n=== 다른 일정 ===")
            for event in sorted(unrelated, key=lambda item: (item.get("time", {}).get("start") or "", item.get("title", ""))):
                print(f"  [{event['event_id']}] {event['title']}")


def _ai_review_list(store: JsonStore) -> None:
    events = [event for event in store.events() if event.get("workflow_status") != "ignored"]
    sources = {source["source_id"]: source for source in store.sources()}
    print("1. 전체  2. Gmail  3. Notion  4. 충돌 없음")
    filter_choice = input("필터: ").strip()
    if filter_choice == "2":
        events = [event for event in events if any(sources.get(ref, {}).get("platform") == "email" for ref in event.get("source_refs", []))]
    elif filter_choice == "3":
        events = [event for event in events if any(sources.get(ref, {}).get("platform") == "notion" for ref in event.get("source_refs", []))]
    elif filter_choice == "4":
        events = [event for event in events if event.get("workflow_status") in {"duplicate_review", "ready", "confirmed"}]
    elif filter_choice != "1":
        print("필터 번호가 올바르지 않습니다.")
        return

    issues = store.review_issues()
    travel_estimates = store.travel_estimates()
    if not events:
        print("해당 조건의 일정이 없습니다.")
        return
    for event in events:
        index = next(i for i, original in enumerate(store.events(), 1) if original["event_id"] == event["event_id"])
        origin = ", ".join(sources.get(ref, {}).get("platform", "") for ref in event.get("source_refs", []) if sources.get(ref)) or "-"
        confidence = event.get("confidence")
        confidence_text = f"{confidence:.0%}" if isinstance(confidence, (int, float)) else "미제공"
        event_issues = [issue["message"] for issue in issues if event["event_id"] in issue.get("event_ids", [])]
        event_id = event["event_id"]
        routes = [
            f"{item['minutes']}분 ({item['mode']})" for item in travel_estimates
            if event_id in {item["from_event_id"], item["to_event_id"]}
        ]
        travel_text = ", ".join(routes) if routes else "미설정"
        print(f"\n[{index}] {event['title']} | {origin} | 신뢰도 {confidence_text} | {event.get('workflow_status')}")
        print(f"  {event.get('time', {}).get('start') or event.get('time', {}).get('date') or '날짜 미정'} | {(event.get('location') or {}).get('name', '-')} | 이동 {travel_text} | 준비 {event.get('preparation_minutes', 0)}분")
        for message in event_issues:
            print(f"  이슈: {message}")


def _inbox(store: JsonStore) -> None:
    sources = store.sources()
    if not sources:
        print("Inbox 원문이 없습니다. JSON 일정 데이터를 먼저 가져오세요.")
        return
    print("1. 전체  2. Gmail  3. Notion")
    filter_choice = input("필터: ").strip()
    if filter_choice in {"2", "3"}:
        platform = "email" if filter_choice == "2" else "notion"
        sources = [source for source in sources if source.get("platform") == platform]
    elif filter_choice != "1":
        print("필터 번호가 올바르지 않습니다.")
        return
    if not sources:
        print("해당 서비스의 원문이 없습니다.")
        return

    for index, source in enumerate(sources, 1):
        platform = "Gmail" if source.get("platform") == "email" else source.get("platform", "기타")
        print(f"[{index}] {platform} | {source.get('subject') or source.get('reference')} | {source.get('inbox_status', 'detected')} | {source.get('captured_at', '-')}")
    raw = input("미리 볼 원문 번호 (Enter=목록으로 돌아가기): ").strip()
    if not raw:
        return
    index = int(raw)
    if index < 1 or index > len(sources):
        raise ValueError("목록에 표시된 원문 번호를 선택하세요.")
    source = sources[index - 1]
    print("\n=== 원문 미리보기 ===")
    for key in ("platform", "sender", "subject", "reference", "captured_at", "url"):
        if source.get(key):
            print(f"{key}: {source[key]}")
    print(source.get("raw_text") or "원문 본문이 JSON에 없습니다. 현재는 제목·발신자·참조 ID 등 메타데이터만 확인할 수 있습니다.")
    linked = [event for event in store.events() if source["source_id"] in event.get("source_refs", [])]
    for event in linked:
        index = next(i for i, original in enumerate(store.events(), 1) if original["event_id"] == event["event_id"])
        print(f"감지 일정: [{index}] {event['title']} | {event.get('workflow_status')}")


def _ignore_candidate(store: JsonStore) -> None:
    candidates = [
        (index, event) for index, event in enumerate(store.events(), 1)
        if event.get("workflow_status") not in {"ignored", "confirmed"}
    ]
    if not candidates:
        print("무시 처리할 검토 대기 일정이 없습니다.")
        return
    for index, event in candidates:
        print(f"[{index}] {event['title']} | {event.get('workflow_status')} | {event.get('event_id')}")
    index = int(input("무시할 일정 번호: ").strip())
    if index not in {item[0] for item in candidates}:
        raise ValueError("검토 대기 목록에 표시된 번호를 선택하세요.")
    answer = input("이 후보를 무시하고 Inbox 처리를 완료할까요? [y/N]: ").strip().lower()
    if answer == "y":
        event = store.ignore_event(index)
        print(f"무시 처리했습니다: {event['title']}")
    else:
        print("무시 처리를 취소했습니다.")


def _edit_event(store: JsonStore, index: int) -> bool:
    events = store.events()
    if index < 1 or index > len(events):
        raise ValueError(f"일정 번호가 올바르지 않습니다: {index}")
    event = events[index - 1]
    print(f"\n[{index}] {event['title']} 수정. 빈칸은 유지, q는 취소입니다.")
    updates: dict[str, Any] = {}

    title = input(f"일정명 [{event['title']}]: ").strip()
    if title.lower() == "q":
        return False
    if title:
        updates["title"] = title

    print("유형: meeting / personal_appointment / business_meeting / other")
    event_type = input(f"유형 [{event['type']}]: ").strip()
    if event_type.lower() == "q":
        return False
    if event_type:
        if event_type not in {"meeting", "personal_appointment", "business_meeting", "other"}:
            raise ValueError("유형 값이 올바르지 않습니다.")
        updates["type"] = event_type

    organization = input(f"소속 [{event.get('organization') or ''}]: ").strip()
    if organization.lower() == "q":
        return False
    if organization:
        updates["organization"] = organization

    people = input("함께하는 사람 이름(쉼표 구분, 빈칸=유지): ").strip()
    if people.lower() == "q":
        return False
    if people:
        updates["people"] = [{"name": name.strip(), "affiliation": None} for name in people.split(",") if name.strip()]

    current_time = event["time"]
    print("시간은 시간대 오프셋이 포함된 ISO 형식으로 입력하세요. 예: 2026-12-03T15:00:00+09:00")
    start = input(f"시작 [{current_time.get('start') or ''}]: ").strip()
    if start.lower() == "q":
        return False
    end = input(f"종료 [{current_time.get('end') or ''}]: ").strip()
    if end.lower() == "q":
        return False
    time_updates: dict[str, Any] = {}
    if start:
        parsed = datetime.fromisoformat(start.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("시작 시각에는 +09:00 같은 시간대 오프셋을 넣어주세요.")
        time_updates["start"] = start
        time_updates["date"] = parsed.date().isoformat()
    if end:
        parsed = datetime.fromisoformat(end.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("종료 시각에는 +09:00 같은 시간대 오프셋을 넣어주세요.")
        time_updates["end"] = end
        time_updates.setdefault("date", parsed.date().isoformat())
    if time_updates:
        updates["time"] = time_updates

    content = input(f"내용 [{event.get('content') or ''}]: ").strip()
    if content.lower() == "q":
        return False
    if content:
        updates["content"] = content

    preparation = input(f"준비 시간(분) [{event.get('preparation_minutes', '')}]: ").strip()
    if preparation.lower() == "q":
        return False
    if preparation:
        updates["preparation_minutes"] = int(preparation)

    current_location = event.get("location") or {}
    location_name = input(f"장소명 [{current_location.get('name', '')}]: ").strip()
    if location_name.lower() == "q":
        return False
    location_address = input(f"주소 [{current_location.get('address', '')}]: ").strip()
    if location_address.lower() == "q":
        return False
    if location_name or location_address:
        new_location = dict(current_location)
        if location_name:
            new_location["name"] = location_name
        if location_address:
            new_location["address"] = location_address
        updates["location"] = new_location

    if not updates:
        print("변경 사항이 없습니다.")
        return False
    store.update_event(index, updates)
    print("수정 내용을 저장했습니다. 충돌 검사는 다시 진행됩니다.")
    return True


def _choose_event_index(store: JsonStore, event_ids: list[str]) -> int:
    indices = _index_by_id(store)
    options = [(indices[event_id], event_id) for event_id in event_ids if event_id in indices]
    if not options:
        raise ValueError("수정 가능한 일정 후보를 찾지 못했습니다.")
    events = store.events()
    if len(options) == 1:
        index, event_id = options[0]
        print(f"수정 대상으로 [{index}] {events[index - 1]['title']}을(를) 선택합니다.")
        return index

    print("수정할 우리 일정:")
    for index, event_id in options:
        print(f"  [{index}] {events[index - 1]['title']} ({event_id})")
    while True:
        answer = input("수정할 일정 번호: ").strip()
        try:
            index = int(answer)
        except ValueError:
            print("일정 번호를 입력하세요.")
            continue
        if any(index == item[0] for item in options):
            return index
        print("위에 표시된 번호 중에서 선택하세요.")


def _change_event_time(store: JsonStore, index: int) -> bool:
    event = store.events()[index - 1]
    current = event["time"]
    print(f"\n[{index}] {event['title']}의 시간만 수정합니다.")
    print("시간대 오프셋이 포함된 ISO 형식으로 입력하세요. 예: 2026-12-03T15:00:00+09:00")
    start = input(f"시작 [{current.get('start') or ''}] (빈칸=유지, q=취소): ").strip()
    if start.lower() == "q":
        return False
    end = input(f"종료 [{current.get('end') or ''}] (빈칸=유지, q=취소): ").strip()
    if end.lower() == "q":
        return False
    updates: dict[str, str] = {}
    if start:
        parsed = datetime.fromisoformat(start.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("시작 시각에 +09:00 같은 시간대 오프셋이 필요합니다.")
        updates["start"] = start
        updates["date"] = parsed.date().isoformat()
    if end:
        parsed = datetime.fromisoformat(end.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("종료 시각에 +09:00 같은 시간대 오프셋이 필요합니다.")
        updates["end"] = end
        updates.setdefault("date", parsed.date().isoformat())
    if not updates:
        print("시간을 바꾸지 않았습니다.")
        return False
    store.update_event(index, {"time": updates})
    print("시간을 저장했습니다. 다시 충돌 검사를 합니다.")
    return True


def _resolve_time_conflict(store: JsonStore, issue: dict[str, Any]) -> bool:
    current = store.build_input("conflict-choice")
    event_map = {event["event_id"]: event for event in current["event_candidates"]}
    index_map = _index_by_id(store)
    conflict_ids = issue.get("event_ids", [])
    sides = [event_map[event_id] for event_id in conflict_ids if event_id in event_map]
    sides.sort(key=lambda event: event.get("time", {}).get("start") or event.get("start") or "")
    if len(sides) != 2:
        raise ValueError("충돌한 두 일정을 찾지 못했습니다.")

    print("둘 중 어떤 일정을 바꾸시겠습니까?")
    for side_number, event in enumerate(sides, 1):
        event_id = event["event_id"]
        local_index = index_map.get(event_id)
        start = event.get("time", {}).get("start") or event.get("start") or "시간 미정"
        if local_index is None:
            raise ValueError("내부 일정이 아닌 항목이 충돌 목록에 포함되어 있습니다.")
        print(f"  {side_number}. {event['title']} | {start} | 우리 일정 인덱스 {local_index}")

    while True:
        answer = input("수정할 일정 번호 [1/2, q=중단]: ").strip().lower()
        if answer == "q":
            return False
        if answer not in {"1", "2"}:
            print("1 또는 2를 입력하세요.")
            continue
        chosen = sides[int(answer) - 1]
        local_index = index_map.get(chosen["event_id"])
        if local_index is None:
            raise ValueError("내부 일정이 아닌 항목은 수정할 수 없습니다.")
        break

    print("1. 시간만 변경")
    print("2. 이 일정 삭제")
    action = input("선택 [1/2, q=중단]: ").strip().lower()
    if action == "q":
        return False
    if action == "1":
        return _change_event_time(store, local_index)
    if action == "2":
        event = store.events()[local_index - 1]
        answer = input(f"'{event['title']}'을(를) 우리 일정 저장소에서 삭제할까요? [y/N]: ").strip().lower()
        if answer != "y":
            print("삭제를 취소했습니다.")
            return False
        store.delete_event(local_index)
        print("일정을 우리 저장소에서 삭제했습니다. 원본 입력 JSON은 변경하지 않았습니다.")
        return True
    print("1 또는 2를 선택하세요.")
    return False


def _review_conflicts(store: JsonStore, model: str, ollama_url: str) -> bool:
    print("\n[1단계] 시간 충돌·시간 누락 확인")
    if not any(event.get("workflow_status") != "ignored" for event in store.events()):
        store.save_review_issues([])
        print("검토할 일정 후보가 없습니다. 모든 후보가 무시 처리되어 있습니다.")
        return True
    for attempt in range(1, 101):
        result = _run(store, use_llm=False, model=model, ollama_url=ollama_url)
        issues = [issue for issue in result["issues"] if issue["type"] in {"time_conflict", "missing_information"}]
        store.save_review_issues(issues)
        if not issues:
            print("시간 겹침과 필수 시간 누락이 없습니다. 이동시간은 이 단계에서 확인하지 않습니다.")
            return True
        issue = issues[0]
        print(f"\n검토 {attempt}: {issue['message']}")
        if issue["type"] == "time_conflict":
            if _resolve_time_conflict(store, issue):
                continue
            return False
        if issue["type"] == "missing_information":
            index = _choose_event_index(store, issue.get("event_ids", []))
            if _change_event_time(store, index):
                continue
            return False
        index_map = _index_by_id(store)
        for event_id in issue.get("event_ids", []):
            if event_id in index_map:
                event = store.events()[index_map[event_id] - 1]
                print(f"  우리 일정 [{index_map[event_id]}] {event['title']} ({event_id})")
        editable = issue.get("event_ids", [])
        if not editable:
            print("수정 가능한 우리 일정이 없습니다.")
            return False
        index = _choose_event_index(store, editable)
        if not _edit_event(store, index):
            return False
    print("수정 반복 횟수 한도에 도달했습니다. 저장은 유지했으니 다시 실행해 주세요.")
    return False


def _review_duplicates(store: JsonStore, model: str, ollama_url: str) -> bool:
    print("\n[중복 후보 분석]")
    if not any(event.get("workflow_status") != "ignored" for event in store.events()):
        store.save_review_issues([])
        print("중복 검토할 일정 후보가 없습니다.")
        return True
    for attempt in range(1, 101):
        result = _run(store, use_llm=True, model=model, ollama_url=ollama_url)
        issues = [issue for issue in result["issues"] if issue["type"] == "possible_duplicate"]
        store.save_review_issues(issues)
        if not issues:
            if result["diagnostics"]["llm_status"] == "unavailable":
                print("Ollama에 연결하지 못해 중복 검토를 완료하지 못했습니다. Ollama를 켠 뒤 8번 중복 검사를 다시 실행하세요.")
                return False
            print(f"중복 후보가 더 없습니다. Ollama 상태: {result['diagnostics']['llm_status']}")
            return True
        issue = issues[0]
        ids = issue.get("event_ids", [])
        index_map = _index_by_id(store)
        titles: list[tuple[str, str]] = []
        active = {event["event_id"]: event for event in store.events()}
        for event_id in ids:
            event = active.get(event_id)
            if event is not None and event_id in index_map:
                titles.append((event_id, f"[{index_map[event_id]}] {event['title']}"))
        print(f"\n중복 후보 {attempt}: " + " ↔ ".join(title for _, title in titles))
        print(f"근거: {issue['message']}")
        answer = input("같은 일정인가요? [y=같음 / n=다른 일정 / q=중단]: ").strip().lower()
        if answer == "q":
            return False
        if answer == "n":
            store.set_duplicate_decision(ids[0], ids[1], "different")
            print("별도 일정으로 기록했습니다. 다음 후보를 확인합니다.")
            continue
        if answer != "y":
            print("y 또는 n으로 답해야 다음 단계로 넘어갈 수 있습니다.")
            continue

        candidate_ids = [event_id for event_id in issue.get("event_ids", []) if event_id in index_map]
        if len(candidate_ids) > 1:
            print("남길 일정을 고르세요. 다른 한 건은 중복으로 제거됩니다.")
            for event_id in candidate_ids:
                print(f"  [{index_map[event_id]}] {active[event_id]['title']}")
            keep_index = int(input("남길 일정 번호: ").strip())
            if keep_index not in [index_map[event_id] for event_id in candidate_ids]:
                raise ValueError("후보에 표시된 번호를 선택하세요.")
            keep_id = next(event_id for event_id in candidate_ids if index_map[event_id] == keep_index)
            remove_id = next(event_id for event_id in candidate_ids if event_id != keep_id)
            store.remove_duplicate(index_map[remove_id], keep_id)
        elif candidate_ids:
            remove_id = candidate_ids[0]
            other_id = next((event_id for event_id in ids if event_id != remove_id), "")
            store.remove_duplicate(index_map[remove_id], other_id)
        print("중복본을 제거했습니다. 남은 일정을 다시 검사합니다.")
    print("중복 검토 반복 횟수 한도에 도달했습니다. 저장은 유지했으니 다시 실행해 주세요.")
    return False


def _standalone_duplicate_check(store: JsonStore, model: str, ollama_url: str) -> None:
    if not any(event.get("workflow_status") != "ignored" for event in store.events()):
        print("중복 검사할 일정이 없습니다. 먼저 일정 데이터를 가져오세요.")
        return
    if _review_duplicates(store, model, ollama_url):
        print("중복 검사를 마쳤습니다. 내부 캘린더 등록 상태는 변경하지 않았습니다.")


def _import_and_review(store: JsonStore, model: str, ollama_url: str) -> None:
    source_path = Path(__file__).resolve().parent / "examples" / "workflow.demo.input.json"
    print(f"원본 입력 JSON: {source_path}")
    document = json.loads(source_path.read_text(encoding="utf-8-sig"))
    counts = store.import_document(document)
    print(f"별도 저장소: 신규 {counts['added']}건, 내용 변경 {counts['updated']}건, 이미 기록됨 {counts['existing']}건, 총 {counts['total']}건")
    review_event_ids = counts["review_event_ids"]
    if not review_event_ids:
        print("새로 추가되거나 내용이 바뀐 일정이 없어 기존 기록은 다시 가져오거나 검토하지 않습니다.")
        return
    if not _review_conflicts(store, model, ollama_url):
        print("충돌 검토를 중단했습니다. 수정 내용은 저장되어 있으며, 다시 가져오기를 선택하면 검사를 이어갈 수 있습니다.")
        return
    confirmed_count = store.mark_conflicts_clear(review_event_ids)
    print(f"충돌 검토가 완료되었습니다. {confirmed_count}건을 내부 캘린더에 등록했습니다. 중복 검사는 8번에서 원할 때 실행할 수 있습니다.")


def _edit_saved_event(store: JsonStore, model: str, ollama_url: str) -> None:
    _show_events(store)
    events = store.events()
    if not events:
        return
    index = int(input("수정할 일정 번호: ").strip())
    edited_event_id = events[index - 1]["event_id"] if 1 <= index <= len(events) else None
    if _edit_event(store, index):
        if _review_conflicts(store, model, ollama_url):
            if edited_event_id:
                confirmed_count = store.mark_conflicts_clear([edited_event_id])
            else:
                confirmed_count = 0
            print(f"충돌 재검토가 완료되었습니다. {confirmed_count}건을 내부 캘린더에 등록했습니다.")
        else:
            print("수정 내용은 저장했습니다. 충돌 검토가 끝날 때까지 내부 캘린더 등록을 보류합니다.")


def _menu(store: JsonStore, model: str, ollama_url: str) -> int:
    while True:
        print(f"\n=== 일정 에이전트 ===\n저장 JSON: {store.path.resolve()}")
        print("1. 일정 데이터 가져오기 + 충돌 검토")
        print("2. 저장된 일정 수정")
        print("3. 우리 일정 저장소 초기화")
        print("4. 메인 일정 타임라인 / 현황")
        print("5. AI 일정 검토 목록 (전체/Gmail/Notion/충돌 없음)")
        print("6. Inbox 원문 목록 / 미리보기")
        print("7. 일정 후보 무시 처리")
        print("8. 저장된 일정 중복 검사")
        print("9. 날짜별 AI 일정 요약")
        print("10. 연계 일정·프로젝트 흐름 AI 분석")
        print("0. 종료")
        choice = input("선택: ").strip()
        try:
            if choice == "0":
                print("종료합니다.")
                return 0
            if choice == "1":
                _import_and_review(store, model, ollama_url)
            elif choice == "2":
                _edit_saved_event(store, model, ollama_url)
            elif choice == "3":
                answer = input("우리 저장소의 일정·검토 기록을 모두 제거할까요? [y/N]: ").strip().lower()
                if answer == "y":
                    store.reset()
                    print("우리 측 저장 JSON을 초기화했습니다. 원본 입력 JSON은 변경하지 않았습니다.")
                else:
                    print("초기화를 취소했습니다.")
            elif choice == "4":
                _calendar_timeline(store)
            elif choice == "5":
                _ai_review_list(store)
            elif choice == "6":
                _inbox(store)
            elif choice == "7":
                _ignore_candidate(store)
            elif choice == "8":
                _standalone_duplicate_check(store, model, ollama_url)
            elif choice == "9":
                _summarize_day(store, model, ollama_url)
            elif choice == "10":
                _analyze_event_flows(store, model, ollama_url)
            else:
                print("메뉴 번호를 선택하세요.")
        except (OSError, json.JSONDecodeError, ValueError, KeyError) as exc:
            print(f"처리 중 오류: {exc}")
        input("\n메뉴로 돌아가려면 Enter를 누르세요...")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    args = _parse_args()
    try:
        store = JsonStore(args.store)
        return _menu(store, args.model, args.ollama_url)
    except (OSError, json.JSONDecodeError, ValueError, KeyError) as exc:
        print(f"calendar CLI 오류: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
