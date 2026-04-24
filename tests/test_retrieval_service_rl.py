from __future__ import annotations

from pathlib import Path

import pandas as pd

from services.retrieval_service import RetrievalSearchService


class FakeRLEngine:
    def __init__(self, selected_arm: str = "convenience"):
        self.selected_arm = selected_arm
        self.selected_intents: list[str] = []
        self.logged: list[tuple[str, float, str, str]] = []

    def select_strategy(self, intent_name: str, c_param: float = 0.5) -> str:
        self.selected_intents.append(intent_name)
        return self.selected_arm

    def log_user_feedback(self, arm_name: str, reward: float, intent_name: str, query: str = "") -> None:
        self.logged.append((arm_name, reward, intent_name, query))


class FakeRetrieval:
    def __init__(self):
        self.calls: list[dict[str, object]] = []
        self.index = object()

    def build_or_load_index(self, force_rebuild: bool = False):
        return self.index

    def recommend_keywords(self, **kwargs):
        self.calls.append(kwargs)
        return pd.DataFrame(
            [
                {
                    "business_id": "b1",
                    "name": "Cafe One",
                    "address": "123 Market St",
                    "city": "Philadelphia",
                    "state": "PA",
                    "categories": "Coffee & Tea",
                    "stars": 4.5,
                    "review_count": 120,
                    "similarity": 0.7,
                    "final_score": 0.8,
                    "latitude": 39.95,
                    "longitude": -75.16,
                    "distance_km": 0.7,
                    "price_tier": 2.0,
                    "price_match": 0.8,
                }
            ]
        )


def _build_service(monkeypatch, fake_rl: FakeRLEngine) -> RetrievalSearchService:
    monkeypatch.setattr(RetrievalSearchService, "_create_rl_engine", lambda self: fake_rl)
    service = RetrievalSearchService(repo_root=Path("."))
    service._retrieval = FakeRetrieval()
    service._attach_photo_urls = lambda df: df
    return service


def test_search_applies_rl_preset_and_exposes_meta(monkeypatch) -> None:
    fake_rl = FakeRLEngine(selected_arm="convenience")
    service = _build_service(monkeypatch, fake_rl)

    rows, meta = service.search(query="quick cheap lunch", state="PA", city="Philadelphia")

    call = service._retrieval.calls[-1]
    assert fake_rl.selected_intents == ["intent_quick"]
    assert call["w_distance"] == service._RL_WEIGHT_PRESETS["convenience"]["w_distance"]
    assert call["w_semantic"] == service._RL_WEIGHT_PRESETS["convenience"]["w_semantic"]
    assert meta["rl_applied"] is True
    assert meta["rl_selected_arm"] == "convenience"
    assert "Convenience" in str(meta["rl_strategy_label"])
    assert rows[0]["business_id"] == "b1"


def test_search_manual_override_uses_request_weights(monkeypatch) -> None:
    fake_rl = FakeRLEngine(selected_arm="reputation")
    service = _build_service(monkeypatch, fake_rl)

    _, meta = service.search(
        query="romantic dinner",
        state="PA",
        rl_user_overrode=True,
        w_semantic=0.22,
        w_rating=1.11,
        w_price=0.33,
        w_distance=1.44,
        w_popularity=0.55,
    )

    call = service._retrieval.calls[-1]
    assert fake_rl.selected_intents == []
    assert call["w_semantic"] == 0.22
    assert call["w_rating"] == 1.11
    assert call["w_price"] == 0.33
    assert call["w_distance"] == 1.44
    assert call["w_popularity"] == 0.55
    assert meta["rl_applied"] is False
    assert meta["rl_user_override_active"] is True


def test_search_logs_feedback_events_before_next_selection(monkeypatch) -> None:
    fake_rl = FakeRLEngine(selected_arm="explorer")
    service = _build_service(monkeypatch, fake_rl)

    _, meta = service.search(
        query="sushi near campus",
        state="PA",
        rl_prev_selected_arm="convenience",
        rl_prev_intent_name="intent_quick",
        rl_action_events=[
            {"action": "detail_open", "query_text": "quick lunch"},
            {"action": "refresh", "query_text": "quick lunch"},
            {"action": "ignored", "query_text": "quick lunch"},
        ],
    )

    assert fake_rl.logged == [
        ("convenience", 1.0, "intent_quick", "quick lunch"),
        ("convenience", -0.1, "intent_quick", "quick lunch"),
    ]
    assert meta["rl_feedback_logged"] == 2
    assert fake_rl.selected_intents == ["intent_default"]


def test_search_stays_available_when_rl_engine_is_missing(monkeypatch) -> None:
    fake_rl = FakeRLEngine(selected_arm="explorer")
    service = _build_service(monkeypatch, fake_rl)
    service._rl_engine = None

    _, meta = service.search(
        query="burger downtown",
        state="PA",
        w_semantic=0.4,
        w_rating=0.5,
        w_price=0.6,
        w_distance=0.7,
        w_popularity=0.8,
    )

    call = service._retrieval.calls[-1]
    assert call["w_semantic"] == 0.4
    assert call["w_rating"] == 0.5
    assert meta["rl_applied"] is False
    assert meta["rl_selected_arm"] is None
