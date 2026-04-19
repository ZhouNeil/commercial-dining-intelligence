"""
Multi-Armed Bandit (MAB) Feedback Loop Infrastructure (MVP Core)
Owner:  (Role 6)
Reference: Syllabus Chapter 11 (Reinforcement Learning)

This module manages the exploration-exploitation tradeoff using Epsilon-Greedy
and updates recommendation strategy weights using the Q-Learning formula.
"""

import pandas as pd
import numpy as np
import os
import json

class RLFeedbackLoop:
    def __init__(self, log_path='data/processed_csv/feedback_log.csv', q_path='data/processed_csv/q_values.json'):
        """
        Initializes the RL environment, interaction logger, and Q-value table.
        """
        self.log_path = log_path
        self.q_path = q_path
        self.alpha = 0.15  # Learning Rate: How much new feedback overwrites old beliefs
        
        # [W_sem, W_rat, W_dist]
        self.arms = {
            "explorer": [0.7, 0.1, 0.2],      # Semantic focused
            "reputation": [0.2, 0.7, 0.1],     # Rating focused
            "convenience": [0.1, 0.2, 0.7]     # Distance focused
        }
        
        self._initialize_files()

    def _initialize_files(self):
        """Ensures both the CSV log and the JSON Q-table exist."""
        log_dir = os.path.dirname(self.log_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Initialize CSV Log
        if not os.path.exists(self.log_path):
            headers = ['timestamp', 'arm_selected', 'reward', 'query_text', 'new_q_value']
            pd.DataFrame(columns=headers).to_csv(self.log_path, index=False)
            
        # Initialize Q-Table (Dictionary)
        if not os.path.exists(self.q_path):
            initial_q = {arm: 0.0 for arm in self.arms.keys()}
            with open(self.q_path, 'w') as f:
                json.dump(initial_q, f)
            self.q_values = initial_q
        else:
            with open(self.q_path, 'r') as f:
                self.q_values = json.load(f)

    def select_strategy(self, epsilon=0.2):
        """
        Level 3 Implementation: True Epsilon-Greedy Selection.
        """
        if np.random.rand() < epsilon:
            # 20% Exploration: Pick a random strategy
            chosen_arm = np.random.choice(list(self.arms.keys()))
            print(f"[RL Logic] EXPLORING: randomly chose '{chosen_arm}'")
            return chosen_arm
        else:
            # 80% Exploitation: Pick the strategy with the highest Q-value
            chosen_arm = max(self.q_values, key=self.q_values.get)
            print(f"[RL Logic] EXPLOITING: chose best arm '{chosen_arm}' (Q={self.q_values[chosen_arm]:.3f})")
            return chosen_arm

    def get_strategy_weights(self, arm_name):
        """Returns the specific [W_sem, W_rat, W_dist] for the chosen arm."""
        return self.arms.get(arm_name, [0.33, 0.33, 0.34])

    def log_user_feedback(self, arm_name, reward, query=""):
        """
        Level 3 Implementation: Log interaction AND update Q-Value.
        """
        # 1. Math Update: Q_new = Q_old + alpha * (Reward - Q_old)
        old_q = self.q_values[arm_name]
        new_q = old_q + self.alpha * (reward - old_q)
        self.q_values[arm_name] = new_q
        
        # 2. Persist the updated Q-table to JSON
        with open(self.q_path, 'w') as f:
            json.dump(self.q_values, f, indent=4)
            
        # 3. Log the history to CSV
        interaction_entry = {
            'timestamp': pd.Timestamp.now(),
            'arm_selected': arm_name,
            'reward': reward,
            'query_text': query,
            'new_q_value': round(new_q, 4)
        }
        pd.DataFrame([interaction_entry]).to_csv(self.log_path, mode='a', header=False, index=False)
        print(f"DEBUG: Reward {reward} applied to '{arm_name}'. Q-value updated: {old_q:.3f} -> {new_q:.3f}")

# ==========================================
# Mocking Script: Run this file directly to test the math!
# ==========================================
if __name__ == "__main__":
    print("--- Starting RL Engine Core Test ---")
    rl_engine = RLFeedbackLoop()
    
    print("\nInitial Q-Values:")
    print(json.dumps(rl_engine.q_values, indent=2))
    
    print("\n--- Simulating 10 user interactions (User really wants convenience) ---")
    for i in range(1, 11):
        print(f"\nInteraction #{i}:")
        # System decides which arm to use
        current_arm = rl_engine.select_strategy(epsilon=0.2)
        
        # Simulating User Behavior: 
        # If system shows 'convenience', user clicks it (+1)
        # If system shows anything else, user clicks 'refresh' (-0.1)
        if current_arm == "convenience":
            simulated_reward = 1.0
        else:
            simulated_reward = -0.1
            
        rl_engine.log_user_feedback(current_arm, simulated_reward, query="cafe")
        
    print("\n--- Final Q-Values after 10 interactions ---")
    print(json.dumps(rl_engine.q_values, indent=2))
    print("\nCheck 'data/processed_csv/' to see the generated JSON and CSV files!")