from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from models.rl_feedback_loop import RLFeedbackLoop, classify_query_intent


def test_initialization_creates_default_intent_buckets(tmp_path: Path) -> None:
    log_path = tmp_path / "feedback_log.csv"
    state_path = tmp_path / "q_values.json"

    engine = RLFeedbackLoop(log_path=str(log_path), q_path=str(state_path))

    assert state_path.is_file()
    assert log_path.is_file()
    assert set(engine.state.keys()) == {
        "intent_quick",
        "intent_romantic",
        "intent_default",
    }
    for intent_state in engine.state.values():
        assert intent_state["total_interactions"] == 0
        assert set(intent_state["arms"].keys()) == {
            "explorer",
            "reputation",
            "convenience",
        }
        for arm_state in intent_state["arms"].values():
            assert arm_state == {"q_value": 0.0, "pull_count": 0}


def test_initialization_migrates_legacy_flat_state(tmp_path: Path) -> None:
    log_path = tmp_path / "feedback_log.csv"
    state_path = tmp_path / "q_values.json"
    legacy_state = {
        "explorer": 0.2,
        "reputation": 0.7,
        "convenience": -0.1,
    }
    state_path.write_text(json.dumps(legacy_state), encoding="utf-8")

    engine = RLFeedbackLoop(log_path=str(log_path), q_path=str(state_path))

    assert engine.state["intent_default"]["arms"]["explorer"]["q_value"] == 0.2
    assert engine.state["intent_default"]["arms"]["reputation"]["q_value"] == 0.7
    assert engine.state["intent_default"]["arms"]["convenience"]["q_value"] == -0.1
    assert engine.state["intent_default"]["total_interactions"] == 0
    assert engine.state["intent_quick"]["arms"]["explorer"]["q_value"] == 0.0


def test_select_strategy_force_explores_unpulled_arms(tmp_path: Path) -> None:
    engine = RLFeedbackLoop(
        log_path=str(tmp_path / "feedback_log.csv"),
        q_path=str(tmp_path / "q_values.json"),
    )

    assert engine.select_strategy("intent_quick") == "explorer"
    engine.log_user_feedback("explorer", reward=0.2, intent_name="intent_quick", query="quick lunch")

    assert engine.select_strategy("intent_quick") == "reputation"
    engine.log_user_feedback("reputation", reward=0.1, intent_name="intent_quick", query="quick lunch")

    assert engine.select_strategy("intent_quick") == "convenience"


def test_select_strategy_uses_ucb_after_all_arms_pulled(tmp_path: Path) -> None:
    engine = RLFeedbackLoop(
        log_path=str(tmp_path / "feedback_log.csv"),
        q_path=str(tmp_path / "q_values.json"),
    )
    engine.state["intent_romantic"] = {
        "total_interactions": 9,
        "arms": {
            "explorer": {"q_value": 0.1, "pull_count": 3},
            "reputation": {"q_value": 0.8, "pull_count": 3},
            "convenience": {"q_value": 0.2, "pull_count": 3},
        },
    }

    chosen = engine.select_strategy("intent_romantic", c_param=0.5)

    scores = {
        name: arm["q_value"] + 0.5 * math.sqrt(math.log(9) / arm["pull_count"])
        for name, arm in engine.state["intent_romantic"]["arms"].items()
    }
    assert chosen == max(scores, key=scores.get) == "reputation"


def test_log_user_feedback_updates_correct_intent_and_csv(tmp_path: Path) -> None:
    log_path = tmp_path / "feedback_log.csv"
    state_path = tmp_path / "q_values.json"
    engine = RLFeedbackLoop(log_path=str(log_path), q_path=str(state_path))

    engine.log_user_feedback(
        "convenience",
        reward=1.0,
        intent_name="intent_quick",
        query="cheap grab-and-go lunch",
    )

    quick_arm = engine.state["intent_quick"]["arms"]["convenience"]
    assert engine.state["intent_quick"]["total_interactions"] == 1
    assert quick_arm["pull_count"] == 1
    assert quick_arm["q_value"] == 0.15
    assert engine.state["intent_default"]["total_interactions"] == 0

    saved_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved_state["intent_quick"]["arms"]["convenience"]["pull_count"] == 1

    with log_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["intent_name"] == "intent_quick"
    assert rows[0]["arm_selected"] == "convenience"
    assert rows[0]["reward"] == "1.0"
    assert rows[0]["query_text"] == "cheap grab-and-go lunch"


def test_classify_query_intent_returns_expected_bucket() -> None:
    assert classify_query_intent("need a quick cheap bite") == "intent_quick"
    assert classify_query_intent("cozy anniversary dinner") == "intent_romantic"
    assert classify_query_intent("sushi in philly") == "intent_default"
