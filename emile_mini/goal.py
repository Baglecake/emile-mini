"""
Goal management and reinforcement integration for the QSE-Émile agent:
 - Maintain a set of goals and Q-values
 - Select current goal via epsilon-greedy using QSE-provided exploration rate
 - Update Q-values based on feedback (reward signals)
"""
import random
from .config import CONFIG

class GoalModule:
    def __init__(self, cfg=CONFIG):
        self.cfg = cfg
        # Known goals and their Q-values
        self.goals = []             # list of goal identifiers
        self.q_values = {}          # mapping goal_id -> value
        self.current_goal = None    # the goal selected this cycle
        # Learning rate for Q-value updates
        self.learning_rate = 0.1    # can be tuned or moved to config
        # History of (goal, reward, updated_Q)
        self.history = []

    def add_goal(self, goal_id):
        """Register a new goal in the agent."""
        if goal_id not in self.goals:
            self.goals.append(goal_id)
            self.q_values[goal_id] = 0.0

    def select_action(self, qse_metrics: dict):
        """
        Choose a goal this cycle, balancing exploration vs exploitation.
        Exploration probability is derived from QSE's normalized entropy.
        """
        # Default exploration if not provided
        exploration = qse_metrics.get('normalized_entropy', 0.5)
        # Epsilon-greedy: explore with prob=exploration
        if random.random() < exploration or not self.goals:
            # Explore: random goal
            self.current_goal = random.choice(self.goals) if self.goals else None
        else:
            # Exploit: pick goal(s) with max Q-value
            max_q = max(self.q_values[g] for g in self.goals)
            best = [g for g in self.goals if self.q_values[g] == max_q]
            self.current_goal = random.choice(best)
        return self.current_goal

    def feedback(self, reward: float):
        """
        Receive a scalar reward for the current goal and update its Q-value.
        Uses simple incremental update: Q <- Q + alpha * (reward - Q)
        """
        if self.current_goal is None:
            return
        old_q = self.q_values.get(self.current_goal, 0.0)
        # Temporal difference-like update
        new_q = old_q + self.learning_rate * (reward - old_q)
        self.q_values[self.current_goal] = new_q
        # Record history
        self.history.append({
            'goal': self.current_goal,
            'reward': reward,
            'Q_value': new_q
        })

    def get_q_values(self):
        """Return current Q-values mapping."""
        return dict(self.q_values)

    def get_history(self):
        """Return the history of goal selections and feedback."""

        return list(self.history)
