from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Protocol


class LLMClient(Protocol):
    def is_available(self) -> bool:
        ...

    def extract_events(
        self,
        user_id: str,
        session_id: str,
        timestamp: str,
        week_index: int,
        text: str,
    ) -> List[Dict[str, object]]:
        ...

    def complete_json(self, system_prompt: str, user_prompt: str) -> Dict[str, object]:
        ...


class NoOpLLMClient:
    def is_available(self) -> bool:
        return False

    def extract_events(
        self,
        user_id: str,
        session_id: str,
        timestamp: str,
        week_index: int,
        text: str,
    ) -> List[Dict[str, object]]:
        return []

    def complete_json(self, system_prompt: str, user_prompt: str) -> Dict[str, object]:
        return {}


@dataclass(frozen=True)
class LiteLLMConfig:
    model: str
    temperature: float = 0.0
    max_tokens: int = 1200
    api_base: Optional[str] = None


class LiteLLMClient:
    def __init__(self, config: LiteLLMConfig):
        self.config = config
        try:
            from litellm import completion  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "litellm is not installed. Install dependencies from requirements.txt first."
            ) from exc
        self._completion = completion

    def is_available(self) -> bool:
        return True

    def complete_json(self, system_prompt: str, user_prompt: str) -> Dict[str, object]:
        kwargs = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        if self.config.api_base:
            kwargs["api_base"] = self.config.api_base

        response = self._completion(**kwargs)
        text = response.choices[0].message.content or "{}"
        return _safe_json_loads(text)

    def extract_events(
        self,
        user_id: str,
        session_id: str,
        timestamp: str,
        week_index: int,
        text: str,
    ) -> List[Dict[str, object]]:
        system_prompt = (
            "You extract atomic health timeline events from user text. "
            "Return strict JSON object with key `events`. "
            "Each event must include `label` and `event_type` using one of: "
            "symptom, lifestyle, diet, medication, sleep, stress, work, "
            "improvement, worsening, intervention, exercise, unknown."
        )
        user_prompt = (
            f"user_id={user_id}\n"
            f"session_id={session_id}\n"
            f"timestamp={timestamp}\n"
            f"week_index={week_index}\n"
            f"text={text}\n"
            "Return concise event labels."
        )
        parsed = self.complete_json(system_prompt, user_prompt)
        events = parsed.get("events", [])
        if not isinstance(events, list):
            return []
        out: List[Dict[str, object]] = []
        for event in events:
            if isinstance(event, dict):
                out.append(event)
        return out


def _safe_json_loads(raw_text: str) -> Dict[str, object]:
    raw_text = raw_text.strip()
    if not raw_text:
        return {}
    try:
        loaded = json.loads(raw_text)
        return loaded if isinstance(loaded, dict) else {}
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        try:
            loaded = json.loads(raw_text[start : end + 1])
            return loaded if isinstance(loaded, dict) else {}
        except json.JSONDecodeError:
            return {}


def build_llm_client(
    provider: str = "none",
    model: Optional[str] = None,
    api_base: Optional[str] = None,
) -> LLMClient:
    normalized_provider = provider.strip().lower()
    if normalized_provider in {"none", "off", "disabled"}:
        return NoOpLLMClient()

    if normalized_provider in {"litellm", "openai", "anthropic", "gemini", "openrouter"}:
        resolved_model = model or os.getenv("LLM_MODEL")
        if not resolved_model:
            raise ValueError(
                "Model is required for LLM provider. Set --model or LLM_MODEL."
            )
        config = LiteLLMConfig(model=resolved_model, api_base=api_base or os.getenv("LLM_API_BASE"))
        return LiteLLMClient(config)

    raise ValueError(
        f"Unsupported provider `{provider}`. Supported: none, litellm/openai/anthropic/gemini/openrouter."
    )
