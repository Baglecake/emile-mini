
"""
experiments.py - Dedicated experimental runner for QSE-Émile

This module provides clean experimental setups without patching existing code.
Run different experiments to explore the cognitive architecture.
"""

import numpy as np
from collections import Counter
from config import CONFIG
from agent import EmileAgent
from viz import plot_surplus_sigma, plot_context_timeline, plot_goal_timeline


class ExperimentRunner:
    """Clean experimental runner for QSE-Émile cognitive architecture"""

    def __init__(self, config=CONFIG):
        self.config = config

    def create_agent_with_goals(self, goals=None):
        """Create an agent with predefined goals"""
        if goals is None:
            goals = ["explore", "exploit", "maintain", "adapt", "consolidate"]

        agent = EmileAgent(self.config)
        for goal_id in goals:
            agent.goal.add_goal(goal_id)

        return agent

    def create_reward_schedule(self, steps, pattern="mixed"):
        """Create different reward schedules for learning experiments"""
        rewards = np.zeros(steps)

        if pattern == "periodic":
            # Regular periodic rewards
            for i in range(0, steps, 20):
                rewards[i] = 0.8

        elif pattern == "random":
            # Random environmental rewards
            for i in range(steps):
                if np.random.random() < 0.05:
                    rewards[i] = np.random.uniform(0.3, 0.9)
                elif np.random.random() < 0.02:
                    rewards[i] = -0.3

        elif pattern == "mixed":
            # Mixed pattern - most realistic
            for i in range(steps):
                if i % 25 == 0:  # Periodic exploration reward
                    rewards[i] = 0.8
                elif i % 17 == 0:  # Different period exploitation
                    rewards[i] = 0.6
                elif i % 40 == 0:  # Maintenance reward
                    rewards[i] = 1.0
                elif np.random.random() < 0.03:
                    rewards[i] = np.random.uniform(0.2, 0.7)
                elif np.random.random() < 0.01:
                    rewards[i] = -0.2

        elif pattern == "none":
            # No rewards - test intrinsic dynamics
            pass

        return rewards

    def run_experiment(self, name, steps=300, dt=0.01, goals=None, reward_pattern="mixed", verbose=True):
        """Run a single experiment with specified parameters"""

        if verbose:
            print(f"\n=== Running Experiment: {name} ===")
            print(f"Steps: {steps}, dt: {dt}")
            print(f"Reward pattern: {reward_pattern}")

        # Create agent with goals
        agent = self.create_agent_with_goals(goals)
        if verbose:
            print(f"Goals: {agent.goal.goals}")
            print(f"Initial Q-values: {agent.goal.q_values}")

        # Create reward schedule
        reward_schedule = self.create_reward_schedule(steps, reward_pattern)
        total_reward = np.sum(np.abs(reward_schedule))
        if verbose:
            print(f"Total reward available: {total_reward:.2f}")

        # Run simulation
        for t in range(steps):
            external_input = None
            if reward_schedule[t] != 0.0:
                external_input = {'reward': reward_schedule[t]}

            agent.step(dt=dt, external_input=external_input)

        # Get results
        history = agent.get_history()
        final_q_values = agent.goal.get_q_values()

        if verbose:
            print(f"Final Q-values: {final_q_values}")
            self._analyze_results(history, final_q_values)

        return {
            'name': name,
            'history': history,
            'q_values': final_q_values,
            'agent': agent,
            'config': self.config
        }

    def _analyze_results(self, history, q_values):
        """Analyze and print experiment results"""

        # Basic dynamics
        surplus = np.array(history['surplus_mean'])
        sigma = np.array(history['sigma_mean'])
        contexts = np.array(history['context_id'])
        goals = history['goal']

        print(f"\n--- Results Analysis ---")
        print(f"Surplus: mean={surplus.mean():.3f}, std={surplus.std():.3f}")
        print(f"Sigma: mean={sigma.mean():.3f}, std={sigma.std():.3f}")
        print(f"Context shifts: {len(set(contexts))}")

        # Goal analysis
        active_goals = [g for g in goals if g is not None]
        if active_goals:
            goal_counts = Counter(active_goals)
            print(f"Goal distribution: {dict(goal_counts)}")

            # Q-value rankings
            sorted_q = sorted(q_values.items(), key=lambda x: x[1], reverse=True)
            print(f"Q-value ranking: {sorted_q}")
        else:
            print("No goals were selected (check goal system)")

        # Oscillation analysis
        if len(sigma) > 20:
            from scipy.fft import fft
            sigma_fft = np.abs(fft(sigma - sigma.mean()))
            if len(sigma_fft) > 2:
                dom_freq_idx = np.argmax(sigma_fft[1:len(sigma_fft)//2]) + 1
                period = len(sigma) / dom_freq_idx
                print(f"Dominant oscillation period: {period:.1f} steps")

    def compare_experiments(self, experiments):
        """Compare results across multiple experiments"""
        print(f"\n=== EXPERIMENT COMPARISON ===")

        for exp in experiments:
            name = exp['name']
            history = exp['history']
            q_vals = exp['q_values']

            surplus_mean = np.mean(history['surplus_mean'])
            sigma_std = np.std(history['sigma_mean'])
            context_shifts = len(set(history['context_id']))

            # Goal diversity
            active_goals = [g for g in history['goal'] if g is not None]
            goal_diversity = len(set(active_goals)) if active_goals else 0

            print(f"{name:15} | surplus={surplus_mean:.3f} | σ_std={sigma_std:.3f} | "
                  f"contexts={context_shifts:2d} | goals={goal_diversity}")

    def plot_experiment(self, experiment, dt=0.01):
        """Plot results from an experiment"""
        history = experiment['history']
        name = experiment['name']

        print(f"\nPlotting experiment: {name}")
        plot_surplus_sigma(history, dt)
        plot_context_timeline(history, dt)
        plot_goal_timeline(history, dt)


def main():
    """Run a suite of experiments to explore the QSE-Émile architecture"""

    runner = ExperimentRunner()
    experiments = []

    print("QSE-Émile Experimental Suite")
    print("=" * 40)

    # Experiment 1: Basic dynamics with mixed rewards
    exp1 = runner.run_experiment(
        name="Basic Mixed",
        steps=200,
        reward_pattern="mixed"
    )
    experiments.append(exp1)

    # Experiment 2: Intrinsic dynamics only (no rewards)
    exp2 = runner.run_experiment(
        name="Intrinsic Only",
        steps=200,
        reward_pattern="none"
    )
    experiments.append(exp2)

    # Experiment 3: Periodic rewards only
    exp3 = runner.run_experiment(
        name="Periodic Rewards",
        steps=200,
        reward_pattern="periodic"
    )
    experiments.append(exp3)

    # Experiment 4: Longer run to see evolution
    exp4 = runner.run_experiment(
        name="Long Evolution",
        steps=500,
        reward_pattern="mixed"
    )
    experiments.append(exp4)

    # Compare all experiments
    runner.compare_experiments(experiments)

    # Plot the most interesting one
    runner.plot_experiment(exp4)

    print(f"\n=== EXPERIMENT SUITE COMPLETE ===")
    print(f"To plot any experiment, use: runner.plot_experiment(exp_name)")

    return experiments


if __name__ == "__main__":
    experiments = main()
