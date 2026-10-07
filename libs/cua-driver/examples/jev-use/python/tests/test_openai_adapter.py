from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from choose_action import validate_request
from core import build_candidates, parse_visual_regions
from decision_models import DecisionRequest, OpenAIDecisionModel, choose
from openai_adapter import (
    DECISIONS_MODEL,
    HttpDecisionsClient,
    choose_bounded_with_openai,
    choose_with_openai,
)

CRITERIA = {
    "submit-form": "Submit the form.",
    "reobserve": "Get a fresh observation.",
    "abstain": "Stop without acting.",
}


def _answer(**overrides) -> dict:
    answer = {
        "type": "choice",
        "name": "driver_action",
        "choice": "submit-form",
        "confidence": 0.9,
        "probabilities": [
            {"value": "submit-form", "probability": 0.9},
            {"value": "reobserve", "probability": 0.05},
            {"value": "abstain", "probability": 0.05},
        ],
    }
    answer.update(overrides)
    return answer


def _response(**overrides) -> dict:
    response = {"model": "gpt-6-luna", "answers": [_answer()]}
    response.update(overrides)
    return response


class FakeDecisionsClient:
    model = "test-decisions-model"

    def __init__(self, response) -> None:
        self.response = response
        self.payloads: list[dict] = []

    def create_decision(self, payload):
        self.payloads.append(json.loads(json.dumps(payload)))
        return self.response


def _request() -> dict:
    return validate_request(
        {
            "schema": "cua.jev_choice_request_v1",
            "goal": "Submit the verified form.",
            "capture_id": "capture-1",
            "regions": [],
            "history": [],
            "candidates": [
                {"id": candidate_id, "description": description}
                for candidate_id, description in CRITERIA.items()
            ],
        }
    )


class BoundedChoiceTest(unittest.TestCase):
    def test_payload_offers_only_supplied_candidates(self) -> None:
        client = FakeDecisionsClient(_response())
        choice = choose_bounded_with_openai(
            client,
            goal="Pick the next action.",
            state={"observation": {"form": {"verification_field": "empty"}}},
            criteria=CRITERIA,
        )
        self.assertEqual(choice.selected_id, "submit-form")
        self.assertEqual(choice.confidence, 0.9)
        self.assertEqual(choice.probabilities["submit-form"], 0.9)
        self.assertEqual(choice.model, "gpt-6-luna")
        payload = client.payloads[0]
        self.assertEqual(payload["model"], "test-decisions-model")
        question = payload["questions"][0]
        self.assertEqual(question["type"], "choice")
        self.assertEqual(question["name"], "driver_action")
        self.assertEqual(question["instructions"], "Pick the next action.")
        self.assertEqual(
            {item["value"]: item["description"] for item in question["choices"]},
            CRITERIA,
        )
        text = payload["input"][0]["content"][0]
        self.assertEqual(text["type"], "input_text")
        self.assertEqual(
            json.loads(text["text"]),
            {"state": {"observation": {"form": {"verification_field": "empty"}}}},
        )
        wire = json.dumps(payload)
        for forbidden in ("browser_click", "screenshot", "capture_id="):
            self.assertNotIn(forbidden, wire)

    def test_refusal_is_an_error_not_a_selection(self) -> None:
        client = FakeDecisionsClient(
            _response(answers=[{"type": "refusal", "name": "driver_action"}])
        )
        result = choose(
            OpenAIDecisionModel(client),
            DecisionRequest.from_validated(_request()),
        )
        self.assertEqual(result.kind, "error")
        self.assertEqual(result.reason, "model_error")
        self.assertIsNone(result.selected_id)

    def test_selection_must_be_a_supplied_id(self) -> None:
        client = FakeDecisionsClient(_response(answers=[_answer(choice="not-offered")]))
        result = choose(
            OpenAIDecisionModel(client),
            DecisionRequest.from_validated(_request()),
        )
        self.assertEqual(result.kind, "error")

    def test_probabilities_outside_the_table_are_rejected(self) -> None:
        client = FakeDecisionsClient(
            _response(
                answers=[
                    _answer(
                        probabilities=[
                            {"value": "submit-form", "probability": 0.9},
                            {"value": "reobserve", "probability": 0.05},
                            {"value": "abstain", "probability": 0.03},
                            {"value": "mystery", "probability": 0.02},
                        ]
                    )
                ]
            )
        )
        result = choose(
            OpenAIDecisionModel(client),
            DecisionRequest.from_validated(_request()),
        )
        self.assertEqual(result.kind, "error")

    def test_missing_candidate_probability_is_rejected(self) -> None:
        client = FakeDecisionsClient(
            _response(
                answers=[
                    _answer(
                        probabilities=[
                            {"value": "submit-form", "probability": 0.95},
                            {"value": "reobserve", "probability": 0.05},
                        ]
                    )
                ]
            )
        )
        result = choose(
            OpenAIDecisionModel(client),
            DecisionRequest.from_validated(_request()),
        )
        self.assertEqual(result.kind, "error")

    def test_non_finite_probability_is_rejected(self) -> None:
        client = FakeDecisionsClient(
            _response(
                answers=[
                    _answer(
                        probabilities=[
                            {"value": "submit-form", "probability": float("nan")},
                            {"value": "reobserve", "probability": 0.05},
                            {"value": "abstain", "probability": 0.95},
                        ]
                    )
                ]
            )
        )
        result = choose(
            OpenAIDecisionModel(client),
            DecisionRequest.from_validated(_request()),
        )
        self.assertEqual(result.kind, "error")

    def test_tied_top_scores_are_not_actionable(self) -> None:
        client = FakeDecisionsClient(
            _response(
                answers=[
                    _answer(
                        probabilities=[
                            {"value": "submit-form", "probability": 0.5},
                            {"value": "reobserve", "probability": 0.5},
                            {"value": "abstain", "probability": 0.0},
                        ]
                    )
                ]
            )
        )
        # A selection that is not the unique argmax must not dispatch an action.
        result = choose(
            OpenAIDecisionModel(client),
            DecisionRequest.from_validated(_request()),
        )
        self.assertEqual(result.kind, "error")
        self.assertIsNone(result.selected_id)

    def test_missing_confidence_uses_selected_probability(self) -> None:
        answer = _answer()
        del answer["confidence"]
        client = FakeDecisionsClient(_response(answers=[answer]))
        result = choose(
            OpenAIDecisionModel(client),
            DecisionRequest.from_validated(_request()),
        )
        self.assertEqual(result.kind, "selected")
        self.assertEqual(result.selected_id, "submit-form")
        self.assertEqual(result.confidence, 0.9)
        self.assertEqual(result.model, "gpt-6-luna")
        self.assertEqual(set(result.probabilities), set(CRITERIA))

    def test_transport_errors_fail_closed(self) -> None:
        class FailingClient:
            model = "test"

            def create_decision(self, payload):
                raise ValueError("connection refused")

        result = choose(
            OpenAIDecisionModel(FailingClient()),
            DecisionRequest.from_validated(_request()),
        )
        self.assertEqual(result.kind, "error")
        self.assertEqual(result.reason, "model_error")


class FixtureTaskChoiceTest(unittest.TestCase):
    def test_task_choice_uses_driver_action_question(self) -> None:
        snapshot = {
            "target_id": "target",
            "tab_id": "tab",
            "page": {"url": "http://fixture.test/"},
            "outline": "textbox verification value",
            "refs": [
                {
                    "role": "textbox",
                    "name": "verification value",
                    "ref": "p1:0",
                    "value": "expected",
                }
            ],
        }
        payload = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/parse-visual-regions-submit-v1.json"
            ).read_text(encoding="utf-8")
        )
        visual = parse_visual_regions(
            payload,
            expected_capture_id="capture-submit",
            expected_pid=7,
            expected_window_id=9,
        )
        candidates = build_candidates(snapshot, "expected", visual, capture_bound_click=True)
        client = FakeDecisionsClient(_response())
        selected, confidence, probabilities = choose_with_openai(
            client, candidates, snapshot, visual, [], "expected"
        )
        self.assertEqual(selected, "submit-form")
        self.assertEqual(confidence, 0.9)
        self.assertEqual(probabilities[selected], 0.9)
        question = client.payloads[0]["questions"][0]
        self.assertEqual(question["name"], "driver_action")
        self.assertEqual(
            {item["value"] for item in question["choices"]},
            {"submit-form", "reobserve", "abstain"},
        )
        state_text = client.payloads[0]["input"][0]["content"][0]["text"]
        state = json.loads(state_text)["state"]
        self.assertEqual(
            state["observation"]["form"]["verification_field"],
            "contains_required_token",
        )
        self.assertNotIn('"expected"', state_text)


class HttpClientTest(unittest.TestCase):
    def test_missing_key_refuses_setup(self) -> None:
        import os

        saved = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with self.assertRaises(ValueError):
                HttpDecisionsClient(api_key="")
        finally:
            if saved is not None:
                os.environ["OPENAI_API_KEY"] = saved

    def test_base_url_must_be_https_off_loopback(self) -> None:
        with self.assertRaises(ValueError):
            HttpDecisionsClient("k", base_url="http://evil.example.com")
        client = HttpDecisionsClient("k", base_url="http://127.0.0.1:8791")
        self.assertEqual(client._url, "http://127.0.0.1:8791/v1/decisions")

    def test_post_shape_and_response_parsing(self) -> None:
        class FakeResponse:
            def __init__(self, body: bytes) -> None:
                self._body = body

            def read(self, *_args) -> bytes:
                return self._body

            def __enter__(self):
                return self

            def __exit__(self, *_args) -> None:
                return None

        seen = {}

        def opener(request, timeout=None):
            seen["url"] = request.full_url
            seen["method"] = request.get_method()
            seen["auth"] = request.get_header("Authorization")
            seen["body"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse(json.dumps(_response()).encode("utf-8"))

        client = HttpDecisionsClient("sk-test", opener=opener)
        response = client.create_decision({"questions": []})
        self.assertEqual(response["answers"][0]["choice"], "submit-form")
        self.assertEqual(seen["url"], "https://api.openai.com/v1/decisions")
        self.assertEqual(seen["method"], "POST")
        self.assertEqual(seen["auth"], "Bearer sk-test")
        self.assertIn("questions", seen["body"])
        self.assertEqual(seen["body"]["model"], DECISIONS_MODEL)

    def test_http_error_is_a_value_error(self) -> None:
        import urllib.error

        def opener(request, timeout=None):
            raise urllib.error.HTTPError(request.full_url, 429, "rate", {}, None)

        client = HttpDecisionsClient("sk-test", opener=opener)
        with self.assertRaises(ValueError):
            client.create_decision({})


if __name__ == "__main__":
    unittest.main()
