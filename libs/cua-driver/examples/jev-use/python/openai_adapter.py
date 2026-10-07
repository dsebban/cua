"""OpenAI Decisions API adapter for the jev-use example.

The OpenAI Decisions endpoint (``POST /v1/decisions``, public beta) evaluates
shared evidence against typed questions. A ``choice`` question selects one of
the caller's supplied options and reports a per-option probability, which is
exactly the bounded-candidate contract this example uses for TypeSafe Jev.

Like the Jev adapter, this module lives outside Driver: the caller builds the
immutable candidate table and sends only the goal/state text plus candidate
IDs and descriptions. The response contains no tool names, arguments,
coordinates, or screenshot bytes, and the caller still resolves the selected
ID against its own candidate table before dispatching any Driver action.

The API key comes from ``OPENAI_API_KEY`` and is never sent anywhere else.
``OPENAI_BASE_URL`` may point at an OpenAI-compatible gateway; non-loopback
``http://`` URLs are refused.
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Mapping, Protocol

from core import Candidate
from jev_adapter import (
    DRIVER_ACTION_INSTRUCTIONS,
    ProviderChoice,
    _candidate_criteria,
    task_decision_state,
)
from tasks import FixtureFormTask, Task, TaskSources, fixture_sources
from core import VisualObservation

DECISIONS_MODEL = "gpt-6-luna"
DECISIONS_PATH = "/v1/decisions"
DEFAULT_BASE_URL = "https://api.openai.com"
QUESTION_NAME = "driver_action"
_TIMEOUT_SECONDS = 30.0
_MAX_RESPONSE_BYTES = 1 << 20
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}


class DecisionsClient(Protocol):
    """Anything that can submit one Decisions request; fakes use this."""

    model: str

    def create_decision(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...


class HttpDecisionsClient:
    """Minimal stdlib client for ``POST <base>/v1/decisions``.

    ``api_key`` defaults to ``OPENAI_API_KEY``; ``base_url`` defaults to
    ``OPENAI_BASE_URL`` or ``https://api.openai.com``. Plain ``http://`` is
    accepted only for loopback hosts (local gateways, mirrors). ``opener``
    is the ``urllib.request.urlopen`` callable, injectable for tests.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        model: str = DECISIONS_MODEL,
        timeout: float = _TIMEOUT_SECONDS,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        if not isinstance(key, str) or not key.strip():
            raise ValueError("OPENAI_API_KEY is not configured")
        self._auth = "Bearer " + key.strip()
        base = (base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL).strip()
        parsed = urllib.parse.urlsplit(base)
        loopback = parsed.hostname in _LOOPBACK_HOSTS or parsed.netloc in _LOOPBACK_HOSTS
        if parsed.scheme != "https" and not (parsed.scheme == "http" and loopback):
            raise ValueError("OpenAI base URL must be https:// (http:// only on loopback)")
        if not parsed.netloc:
            raise ValueError("OpenAI base URL must include a host")
        self._url = base.rstrip("/") + DECISIONS_PATH
        if not isinstance(model, str) or not 0 < len(model) <= 128:
            raise ValueError("decisions model name must be a bounded string")
        self.model = model
        self.timeout = timeout
        self._opener = opener or urllib.request.urlopen

    def create_decision(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        body = {"model": self.model, **dict(payload)}
        request = urllib.request.Request(
            self._url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": self._auth,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with self._opener(request, timeout=self.timeout) as response:
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as error:
            raise ValueError(f"OpenAI decisions request failed with HTTP {error.code}") from None
        except (OSError, TimeoutError, ValueError) as error:
            raise ValueError(f"OpenAI decisions request failed: {error}") from None
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise ValueError("OpenAI decisions response exceeded the size limit")
        try:
            body = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"OpenAI decisions response was not JSON: {error}") from None
        if not isinstance(body, dict):
            raise ValueError("OpenAI decisions response was not an object")
        return body


def _answer_for(response: Mapping[str, Any], question: str) -> Mapping[str, Any]:
    answers = response.get("answers")
    if not isinstance(answers, list):
        raise ValueError("OpenAI decisions response has no answers")
    matched = [item for item in answers if isinstance(item, dict) and item.get("name") == question]
    if len(matched) != 1:
        raise ValueError(f"OpenAI decisions returned {len(matched)} answers for {question!r}")
    return matched[0]


def _probabilities(answer: Mapping[str, Any], criteria: Mapping[str, str]) -> dict[str, float]:
    raw = answer.get("probabilities")
    if not isinstance(raw, list):
        raise ValueError("OpenAI choice answer has no probabilities")
    probabilities: dict[str, float] = {}
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError("OpenAI probability entry is not an object")
        value, probability = entry.get("value"), entry.get("probability")
        if value not in criteria:
            raise ValueError(f"OpenAI returned a probability for unknown candidate: {value}")
        try:
            score = float(probability)
        except (TypeError, ValueError):
            raise ValueError("OpenAI returned a non-numeric probability") from None
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("OpenAI returned an invalid probability")
        if value in probabilities:
            raise ValueError(f"OpenAI returned a duplicate probability for {value}")
        probabilities[value] = score
    return probabilities


def choose_bounded_with_openai(
    client: DecisionsClient,
    *,
    goal: str,
    state: Mapping[str, Any],
    criteria: Mapping[str, str],
    question: str = QUESTION_NAME,
) -> ProviderChoice:
    """One Decisions ``choice`` question over the caller's candidate table.

    ``state`` is the same secret-redacted decision state the TypeSafe adapter
    sends Jev; it is serialized to JSON as the question's shared evidence.
    """
    if not criteria:
        raise ValueError("OpenAI decisions needs at least one candidate")
    model = getattr(client, "model", None)
    payload = {
        "model": model if isinstance(model, str) and model.strip() else DECISIONS_MODEL,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": json.dumps({"state": dict(state)}, ensure_ascii=False),
                    }
                ],
            }
        ],
        "questions": [
            {
                "type": "choice",
                "name": question,
                "instructions": goal,
                "choices": [
                    {"value": candidate_id, "description": description}
                    for candidate_id, description in criteria.items()
                ],
            }
        ],
    }
    response = client.create_decision(payload)
    answer = _answer_for(response, question)
    if answer.get("type") == "refusal":
        raise ValueError("OpenAI refused the decision question")
    if answer.get("type") != "choice":
        raise ValueError(f"OpenAI returned an unexpected answer type: {answer.get('type')}")
    selected = answer.get("choice")
    if selected not in criteria:
        raise ValueError(f"OpenAI selected unknown candidate: {selected}")
    probabilities = _probabilities(answer, criteria)
    confidence_raw = answer.get("confidence")
    if confidence_raw is None:
        confidence = probabilities.get(selected)
        if confidence is None:
            raise ValueError("OpenAI returned no confidence and no selected probability")
    else:
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            raise ValueError("OpenAI returned a non-numeric confidence") from None
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("OpenAI returned an invalid confidence")
    model_value = response.get("model")
    model_name = model_value if isinstance(model_value, str) and model_value.strip() else None
    return ProviderChoice(selected, confidence, probabilities, model_name)


def choose_for_task_openai(
    client: DecisionsClient,
    task: Task,
    sources: TaskSources,
    candidates: list[Candidate],
    history: list[dict[str, Any]],
) -> tuple[str, float, dict[str, float]]:
    choice = choose_bounded_with_openai(
        client,
        goal=DRIVER_ACTION_INSTRUCTIONS,
        state=task_decision_state(task, sources, history),
        criteria=_candidate_criteria(candidates),
    )
    return choice.selected_id, choice.confidence, choice.probabilities


def choose_live_for_task_openai(
    task: Task,
    sources: TaskSources,
    candidates: list[Candidate],
    history: list[dict[str, Any]],
) -> tuple[str, float, dict[str, float]]:
    return choose_for_task_openai(
        HttpDecisionsClient(), task, sources, candidates, history
    )


def choose_with_openai(
    client: DecisionsClient,
    candidates: list[Candidate],
    snapshot: dict[str, Any],
    visual: VisualObservation | None,
    history: list[dict[str, Any]],
    token: str,
    visual_path: bool = False,
) -> tuple[str, float, dict[str, float]]:
    return choose_for_task_openai(
        client,
        FixtureFormTask(token),
        fixture_sources(snapshot, visual, visual_path=visual_path),
        candidates,
        history,
    )


def choose_live_openai(
    candidates: list[Candidate],
    snapshot: dict[str, Any],
    visual: VisualObservation | None,
    history: list[dict[str, Any]],
    token: str,
    visual_path: bool = False,
) -> tuple[str, float, dict[str, float]]:
    return choose_live_for_task_openai(
        FixtureFormTask(token),
        fixture_sources(snapshot, visual, visual_path=visual_path),
        candidates,
        history,
    )
