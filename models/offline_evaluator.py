"""
Rigorous offline replay evaluator for the contextual UCB RL engine.

This script is intentionally standalone: it generates a reproducible historical
log, replays it against a fresh RLFeedbackLoop instance, and reports baselines,
agent CTR, convergence, and final learned state.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.rl_feedback_loop import DEFAULT_INTENT_BUCKETS, RLFeedbackLoop


# Reproducibility is part of the evaluator contract: every run should regenerate
# the same historical data and the same replay trajectory.
random.seed(42)
np.random.seed(42)

ARMS: tuple[str, ...] = ("convenience", "reputation", "explorer")
GROUND_TRUTH_REWARD_PROBS: dict[str, dict[str, float]] = {
    "intent_quick": {
        "convenience": 0.8,
        "reputation": 0.1,
        "explorer": 0.1,
    },
    "intent_romantic": {
        "convenience": 0.2,
        "reputation": 0.2,
        "explorer": 0.7,
    },
    "intent_default": {
        "convenience": 0.2,
        "reputation": 0.6,
        "explorer": 0.2,
    },
}

DATA_DIR = ROOT / "data" / "processed_csv"
HISTORICAL_LOG_PATH = DATA_DIR / "offline_historical_logs.csv"
OFFLINE_FEEDBACK_LOG_PATH = DATA_DIR / "offline_feedback_log.csv"
OFFLINE_Q_PATH = DATA_DIR / "offline_q_values.json"
WINDOW_SIZE = 100
HISTORICAL_ROWS = 5000


def _ensure_clean_state() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for path in (HISTORICAL_LOG_PATH, OFFLINE_FEEDBACK_LOG_PATH, OFFLINE_Q_PATH):
        if path.exists():
            path.unlink()


def _sample_reward(intent_name: str, arm_name: str) -> int:
    prob = GROUND_TRUTH_REWARD_PROBS[intent_name][arm_name]
    return 1 if random.random() < prob else 0


def _theoretical_ctr(intent_counts: Counter[str], *, optimal: bool) -> float:
    total = sum(intent_counts.values())
    if total <= 0:
        return 0.0
    weighted = 0.0
    for intent_name, count in intent_counts.items():
        probs = GROUND_TRUTH_REWARD_PROBS[intent_name]
        expected = max(probs.values()) if optimal else sum(probs.values()) / len(ARMS)
        weighted += (count / total) * expected
    return float(weighted)


def generate_mock_historical_logs(rows: int = HISTORICAL_ROWS) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for idx in range(rows):
        intent_name = random.choice(DEFAULT_INTENT_BUCKETS)
        historical_arm = random.choice(ARMS)
        reward = _sample_reward(intent_name, historical_arm)
        records.append(
            {
                "row_id": idx,
                "intent_name": intent_name,
                "historical_arm": historical_arm,
                "reward": reward,
                "query_text": f"offline_{intent_name}_{idx}",
            }
        )

    df = pd.DataFrame(records)
    df.to_csv(HISTORICAL_LOG_PATH, index=False)

    intent_counts = Counter(df["intent_name"].astype(str))
    random_ctr = _theoretical_ctr(intent_counts, optimal=False)
    optimal_ctr = _theoretical_ctr(intent_counts, optimal=True)

    print("=== Offline Historical Log ===")
    print(f"Rows generated           : {len(df)}")
    print(f"Theoretical Random CTR   : {random_ctr:.4f}")
    print(f"Theoretical Optimal CTR  : {optimal_ctr:.4f}")
    print()
    return df


def run_replay_evaluation(historical_df: pd.DataFrame) -> dict[str, Any]:
    agent = RLFeedbackLoop(
        log_path=str(OFFLINE_FEEDBACK_LOG_PATH),
        q_path=str(OFFLINE_Q_PATH),
        verbose=False,
    )

    cumulative_reward = 0.0
    matches = 0
    window_rewards: deque[float] = deque(maxlen=WINDOW_SIZE)

    # Replay only updates the agent on policy matches; everything else is rejected.
    for row in historical_df.itertuples(index=False):
        intent_name = str(row.intent_name)
        historical_arm = str(row.historical_arm)
        reward = float(row.reward)
        query_text = str(row.query_text)

        agent_arm = agent.select_strategy(intent_name, c_param=0.5)
        if agent_arm != historical_arm:
            continue

        matches += 1
        cumulative_reward += reward
        window_rewards.append(reward)
        agent.log_user_feedback(agent_arm, reward, intent_name, query=query_text)

    cumulative_ctr = (cumulative_reward / matches) if matches else 0.0
    windowed_ctr = (sum(window_rewards) / len(window_rewards)) if window_rewards else 0.0
    return {
        "agent": agent,
        "matches": matches,
        "cumulative_reward": cumulative_reward,
        "cumulative_ctr": cumulative_ctr,
        "windowed_ctr": windowed_ctr,
    }


def _best_arm_for_intent(intent_state: dict[str, Any]) -> str:
    return max(intent_state["arms"], key=lambda arm_name: float(intent_state["arms"][arm_name]["q_value"]))


def _print_q_state(agent: RLFeedbackLoop) -> None:
    print("=== Final Q-State Verification ===")
    for intent_name in DEFAULT_INTENT_BUCKETS:
        print(f"[{intent_name}]")
        intent_state = agent.state[intent_name]
        print(f"  total_interactions: {intent_state['total_interactions']}")
        for arm_name in ARMS:
            arm_state = intent_state["arms"][arm_name]
            print(
                f"  - {arm_name:<12} pull_count={int(arm_state['pull_count']):>4} "
                f"q_value={float(arm_state['q_value']):.4f}"
            )
        print()


def _assert_expected_winners(agent: RLFeedbackLoop) -> None:
    expected = {
        "intent_quick": "convenience",
        "intent_romantic": "explorer",
        "intent_default": "reputation",
    }
    for intent_name, expected_arm in expected.items():
        actual_arm = _best_arm_for_intent(agent.state[intent_name])
        assert actual_arm == expected_arm, (
            f"{intent_name} expected top arm {expected_arm}, got {actual_arm}. "
            f"State: {json.dumps(agent.state[intent_name], indent=2)}"
        )


def main() -> None:
    _ensure_clean_state()
    historical_df = generate_mock_historical_logs()
    replay = run_replay_evaluation(historical_df)

    intent_counts = Counter(historical_df["intent_name"].astype(str))
    random_ctr = _theoretical_ctr(intent_counts, optimal=False)
    optimal_ctr = _theoretical_ctr(intent_counts, optimal=True)
    agent = replay["agent"]

    print("=== Replay Evaluation Report ===")
    print(f"Theoretical Random CTR       : {random_ctr:.4f}")
    print(f"Theoretical Optimal CTR      : {optimal_ctr:.4f}")
    print(f"Replay matches               : {replay['matches']}")
    print(f"Agent Cumulative CTR         : {replay['cumulative_ctr']:.4f}")
    print(f"Agent Windowed CTR (last 100): {replay['windowed_ctr']:.4f}")
    print()

    _print_q_state(agent)
    _assert_expected_winners(agent)
    print("Assertions passed: each intent bucket learned the expected best arm.")


if __name__ == "__main__":
    main()
