"""Small stdlib-only Ollama client for structured duplicate analysis."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


DUPLICATE_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["left_id", "right_id", "relation", "reason"],
                "properties": {
                    "left_id": {"type": "string"},
                    "right_id": {"type": "string"},
                    "relation": {
                        "type": "string",
                        "enum": ["same_event", "possible_duplicate", "different", "uncertain"],
                    },
                    "reason": {"type": "string"},
                },
            },
        }
    },
    "required": ["pairs"],
}

EVENT_FLOW_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "flows": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["title", "event_ids", "summary", "confidence"],
                "properties": {
                    "title": {"type": "string"},
                    "event_ids": {"type": "array", "items": {"type": "string"}},
                    "summary": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                },
            },
        }
    },
    "required": ["flows"],
}


class OllamaError(RuntimeError):
    """Raised when Ollama cannot return a valid structured response."""


class OllamaClient:
    def __init__(
        self,
        model: str = "qwen3.5:4b",
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: int = 120,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def summarize_schedule(self, day: str, events: list[dict[str, Any]]) -> str:
        """Summarize one day's confirmed internal calendar events in Korean."""
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "options": {"temperature": 0.2},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "당신은 개인 일정 비서입니다. 사용자에게 한국어로 그날 일정을 간결하고 읽기 쉽게 알려주세요. "
                        "입력된 일정에 있는 사실만 사용하고 시간, 장소, 참석자, 내용이 없으면 추측하지 마세요. "
                        "시간순으로 정리하고, 일정 간 이동이나 준비에 주의할 점이 데이터에 명시된 경우에만 언급하세요. "
                        "일정이 많으면 중요한 흐름을 짧게 요약한 뒤 일정 목록을 제시하세요."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps({"date": day, "events": events}, ensure_ascii=False),
                },
            ],
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
            message = body.get("message", {})
            content = message.get("content", "")
            if not isinstance(content, str) or not content.strip():
                raise OllamaError("Ollama returned an empty schedule summary.")
            return content.strip()
        except OllamaError:
            raise
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OllamaError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise OllamaError(f"Could not get a schedule summary from Ollama: {exc}") from exc

    def analyze_event_flows(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Find evidence-based links and chronological flows across calendar events."""
        if len(events) < 2:
            return []
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "format": EVENT_FLOW_RESPONSE_SCHEMA,
            "options": {"temperature": 0},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "당신은 일정 데이터에서 프로젝트와 후속 업무의 흐름을 찾는 분석가입니다. "
                        "입력된 제목, 조직, 참석자, 내용, 연결 원문만 근거로 실제 연관이 있는 일정들을 묶으세요. "
                        "킥오프 뒤 실무 미팅이나 진행 점검처럼 명시된 후속 관계, 검진 뒤 치료처럼 앞 일정의 결과에 "
                        "따라 이어지는 관계를 찾으세요. 날짜나 참석자만 같다는 이유로 묶지 마세요. "
                        "관계가 불분명하면 흐름에 포함하지 마세요. 각 흐름에는 실제 입력 event_id만 넣고 시간 순서로 배열하세요. "
                        "흐름은 두 일정 이상일 때만 반환하고, 제목은 프로젝트나 공통 목적을 짧게 설명하세요. "
                        "요약은 관계의 근거와 선후 관계를 간결하게 설명하세요."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps({"events": events}, ensure_ascii=False),
                },
            ],
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
            message = body.get("message", {})
            content = message.get("content", "")
            if not isinstance(content, str) or not content.strip():
                raise OllamaError("Ollama returned an empty event-flow analysis.")
            result = json.loads(content)
            if not isinstance(result, dict):
                raise OllamaError("Ollama event-flow response must be a JSON object.")
            flows = result.get("flows")
            if not isinstance(flows, list):
                raise OllamaError("Ollama response did not contain a 'flows' array.")
            return [flow for flow in flows if isinstance(flow, dict)]
        except OllamaError:
            raise
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OllamaError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise OllamaError(f"Could not analyze event flows with Ollama: {exc}") from exc

    def classify_duplicate_pairs(self, pairs: list[dict[str, Any]]) -> list[dict[str, str]]:
        if not pairs:
            return []

        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "format": DUPLICATE_RESPONSE_SCHEMA,
            "options": {"temperature": 0},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You compare calendar records. Decide whether each pair describes the same real-world "
                        "appointment. Return only the required JSON. Do not infer missing dates or times. "
                        "Use same_event only when evidence is strong, possible_duplicate when it may be the "
                        "same appointment, different for distinct events, and uncertain when evidence is weak. "
                        "A matching date or person alone is not enough to call two events duplicates."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps({"pairs_to_compare": pairs}, ensure_ascii=False),
                },
            ],
        }

        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
            message = body.get("message", {})
            content = message.get("content", "")
            if not isinstance(content, str) or not content.strip():
                thinking = str(message.get("thinking", ""))[:300]
                raise OllamaError(f"Ollama returned empty message.content. thinking excerpt={thinking!r}")
            try:
                result = json.loads(content)
            except json.JSONDecodeError as exc:
                raise OllamaError(f"Ollama returned non-JSON message.content: {content[:500]!r}") from exc
            classified = result.get("pairs")
            if not isinstance(classified, list):
                raise OllamaError("Ollama response did not contain a 'pairs' array.")
            return [item for item in classified if isinstance(item, dict)]
        except OllamaError:
            raise
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OllamaError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise OllamaError(f"Could not get a valid response from Ollama: {exc}") from exc

