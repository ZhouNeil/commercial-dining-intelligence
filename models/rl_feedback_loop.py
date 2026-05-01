"""
Contextual bandit feedback loop infrastructure.

This module keeps a standalone UCB-based policy learner keyed by coarse intent
buckets. It is intentionally not wired into the live tourist retrieval flow yet.
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any

import pandas as pd


DEFAULT_INTENT_BUCKETS: tuple[str, ...] = (
    "intent_quick",
    "intent_romantic",
    "intent_default",
)


def classify_query_intent(query: str) -> str:
    """Map raw query text into one of the default intent buckets."""
    text = str(query or "").strip().lower()
    if not text:
        return "intent_default"

    quick_patterns = (
        r"\bfast\b",
        r"\bquick\b",
        r"\bcheap\b",
        r"\bgrab[\s-]?and[\s-]?go\b",
    )
    romantic_patterns = (
        r"\bdate\b",
        r"\bromantic\b",
        r"\bcozy\b",
        r"\banniversary\b",
    )

    if any(re.search(pattern, text, re.IGNORECASE) for pattern in quick_patterns):
        return "intent_quick"
    if any(re.search(pattern, text, re.IGNORECASE) for pattern in romantic_patterns):
        return "intent_romantic"
    return "intent_default"


class RLFeedbackLoop:
    def __init__(
        self,
        log_path: str = "data/processed_csv/feedback_log.csv",
        q_path: str = "data/processed_csv/q_values.json",
        verbose: bool = False,
    ):
        """
        Initialize logger, persisted state, and static arm definitions.
        """
        self.log_path = log_path
        self.q_path = q_path
        self.alpha = 0.15
        self.verbose = bool(verbose)

        # [W_sem, W_rat, W_dist]
        self.arms = {
            "explorer": [0.7, 0.1, 0.2],
            "reputation": [0.2, 0.7, 0.1],
            "convenience": [0.1, 0.2, 0.7],
        }
        self.default_intents = DEFAULT_INTENT_BUCKETS

        self.state = self._initialize_files()

    def _emit(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _default_arm_state(self) -> dict[str, float | int]:
        return {"q_value": 0.0, "pull_count": 0}

    def _default_intent_state(self) -> dict[str, Any]:
        return {
            "total_interactions": 0,
            "arms": {arm_name: self._default_arm_state() for arm_name in self.arms},
        }

    def _default_state(self) -> dict[str, Any]:
        return {intent_name: self._default_intent_state() for intent_name in self.default_intents}

    def _ensure_parent_dir(self, path: str) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def _migrate_state(self, raw_state: Any) -> dict[str, Any]:
        """
        Normalize persisted JSON into the nested intent schema.

        This handles empty files, partially-written nested structures, and the
        legacy flat arm->q_value dictionary by folding old arm values into
        `intent_default`.
        """
        base = self._default_state()
        if not isinstance(raw_state, dict) or not raw_state:
            return base

        legacy_flat = set(raw_state.keys()).issubset(set(self.arms.keys()))
        if legacy_flat:
            default_state = base["intent_default"]
            for arm_name, q_value in raw_state.items():
                if arm_name not in self.arms:
                    continue
                try:
                    q_num = float(q_value)
                except (TypeError, ValueError):
                    q_num = 0.0
                default_state["arms"][arm_name] = {"q_value": q_num, "pull_count": 0}
            return base

        for intent_name in self.default_intents:
            intent_raw = raw_state.get(intent_name, {})
            if not isinstance(intent_raw, dict):
                intent_raw = {}
            total = intent_raw.get("total_interactions", 0)
            try:
                total_int = max(0, int(total))
            except (TypeError, ValueError):
                total_int = 0
            base[intent_name]["total_interactions"] = total_int

            arms_raw = intent_raw.get("arms", {})
            if not isinstance(arms_raw, dict):
                arms_raw = {}
            for arm_name in self.arms:
                arm_raw = arms_raw.get(arm_name, {})
                if not isinstance(arm_raw, dict):
                    arm_raw = {}
                q_value = arm_raw.get("q_value", 0.0)
                pull_count = arm_raw.get("pull_count", 0)
                try:
                    q_num = float(q_value)
                except (TypeError, ValueError):
                    q_num = 0.0
                try:
                    pulls = max(0, int(pull_count))
                except (TypeError, ValueError):
                    pulls = 0
                base[intent_name]["arms"][arm_name] = {
                    "q_value": q_num,
                    "pull_count": pulls,
                }
        return base

    def _load_state_from_disk(self) -> dict[str, Any]:
        if not os.path.exists(self.q_path):
            return self._default_state()
        try:
            with open(self.q_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError, ValueError):
            raw = {}
        return self._migrate_state(raw)

    def _save_state(self) -> None:
        with open(self.q_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=4)

    def _initialize_files(self) -> dict[str, Any]:
        """Ensure the CSV log and nested JSON state exist and are normalized."""
        self._ensure_parent_dir(self.log_path)
        self._ensure_parent_dir(self.q_path)

        if not os.path.exists(self.log_path):
            headers = [
                "timestamp",
                "intent_name",
                "arm_selected",
                "reward",
                "query_text",
                "new_q_value",
            ]
            pd.DataFrame(columns=headers).to_csv(self.log_path, index=False)

        state = self._load_state_from_disk()
        self.state = state
        self._save_state()
        return state

    def _intent_state(self, intent_name: str) -> dict[str, Any]:
        name = str(intent_name or "").strip() or "intent_default"
        if name not in self.state:
            self.state[name] = self._default_intent_state()
            self._save_state()
        return self.state[name]

    def select_strategy(self, intent_name: str, c_param: float = 0.5) -> str:
        """Choose an arm for the provided intent bucket using UCB."""
        intent_state = self._intent_state(intent_name)
        arms_state = intent_state["arms"]
        # Force-explore any arm with zero pulls before applying UCB scores,
        # which would otherwise compute log(N)/0.
        for arm_name in self.arms:
            if int(arms_state[arm_name]["pull_count"]) == 0:
                self._emit(
                    f"[RL Logic] FORCE-EXPLORE: intent '{intent_name}' chose untried arm '{arm_name}'"
                )
                return arm_name

        total_interactions = max(1, int(intent_state["total_interactions"]))
        log_total = math.log(total_interactions)
        scores: dict[str, float] = {}
        for arm_name in self.arms:
            arm_state = arms_state[arm_name]
            q_value = float(arm_state["q_value"])
            pull_count = max(1, int(arm_state["pull_count"]))
            bonus = float(c_param) * math.sqrt(log_total / pull_count)
            scores[arm_name] = q_value + bonus

        chosen_arm = max(scores, key=scores.get)
        self._emit(
            f"[RL Logic] UCB: intent '{intent_name}' chose '{chosen_arm}' "
            f"(score={scores[chosen_arm]:.3f})"
        )
        return chosen_arm

    def get_strategy_weights(self, arm_name: str) -> list[float]:
        """Return the [W_sem, W_rat, W_dist] weights for a chosen arm."""
        return self.arms.get(arm_name, [0.33, 0.33, 0.34])

    def log_user_feedback(
        self,
        arm_name: str,
        reward: float,
        intent_name: str,
        query: str = "",
    ) -> None:
        """Update the arm state for the provided intent and append a CSV log row."""
        if arm_name not in self.arms:
            raise KeyError(f"Unknown arm: {arm_name}")

        intent_state = self._intent_state(intent_name)
        arm_state = intent_state["arms"][arm_name]
        old_q = float(arm_state["q_value"])
        reward_f = float(reward)
        # TD(0) / exponential moving average: new_q = (1-alpha)*old_q + alpha*reward.
        # alpha=0.15 weights recent rewards at 15% and decays history at 85% per step.
        new_q = old_q + self.alpha * (reward_f - old_q)

        arm_state["q_value"] = new_q
        arm_state["pull_count"] = int(arm_state["pull_count"]) + 1
        intent_state["total_interactions"] = int(intent_state["total_interactions"]) + 1
        self._save_state()

        interaction_entry = {
            "timestamp": pd.Timestamp.now(),
            "intent_name": str(intent_name or "").strip() or "intent_default",
            "arm_selected": arm_name,
            "reward": reward_f,
            "query_text": query,
            "new_q_value": round(new_q, 4),
        }
        pd.DataFrame([interaction_entry]).to_csv(
            self.log_path,
            mode="a",
            header=False,
            index=False,
        )
        self._emit(
            f"DEBUG: Reward {reward_f} applied to '{arm_name}' for "
            f"'{interaction_entry['intent_name']}'. Q-value updated: {old_q:.3f} -> {new_q:.3f}"
        )


if __name__ == "__main__":
    print("--- Starting Contextual UCB RL Engine Demo ---")
    rl_engine = RLFeedbackLoop()

    query = "quick cheap lunch near campus"
    detected_intent = classify_query_intent(query)
    print(f"\nQuery: {query!r}")
    print(f"Detected intent: {detected_intent}")

    print("\nInitial state:")
    print(json.dumps(rl_engine.state, indent=2))

    print("\n--- Simulating 6 user interactions ---")
    for i in range(1, 7):
        print(f"\nInteraction #{i}:")
        current_arm = rl_engine.select_strategy(detected_intent, c_param=0.5)
        simulated_reward = 1.0 if current_arm == "convenience" else -0.1
        rl_engine.log_user_feedback(
            current_arm,
            simulated_reward,
            intent_name=detected_intent,
            query=query,
        )

    print("\n--- Final nested state ---")
    print(json.dumps(rl_engine.state, indent=2))
    print("\nCheck 'data/processed_csv/' to see the generated JSON and CSV files.")
