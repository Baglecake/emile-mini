"""
run_experiments.py - Clean experimental runner for QSE-Émile

Just run: python run_experiments.py
"""

import numpy as np
from collections import Counter
from config import CONFIG
from agent import EmileAgent
from viz import plot_surplus_sigma, plot_context_timeline, plot_goal_timeline


def run_experiment(name, steps, reward_pattern="mixed"):
    """Run one experiment"""
    print(f"\n=== {name} ===")

    # Create agent with goals
    agent = EmileAgent(CONFIG)
    for goal_id in ["explore", "exploit", "maintain", "adapt"]:
        agent.goal.add_goal(goal_id)

    # Create rewards
    for t in range(steps):
        reward = 0.0
        if reward_pattern == "mixed":
            if t % 25 == 0:
                reward = 0.8
            elif t % 40 == 0:
                reward = 0.6
        elif reward_pattern == "periodic":
            if t % 20 == 0:
                reward = 0.8
        # else no rewards

        ext = {'reward': reward} if reward > 0 else None
        agent.step(dt=0.01, external_input=ext)

    # Get results
    history = agent.get_history()

    # Show results
    surplus = np.array(history['surplus_mean'])
    sigma = np.array(history['sigma_mean'])
    goals_selected = [g for g in history['goal'] if g is not None]

    print(f"Surplus: {surplus.mean():.3f} ± {surplus.std():.3f}")
    print(f"Sigma: {sigma.mean():.3f} ± {sigma.std():.3f}")
    print(f"Contexts: {len(set(history['context_id']))}")

    if goals_selected:
        goal_counts = Counter(goals_selected)
        print(f"Goals: {dict(goal_counts)}")
        print(f"Q-values: {agent.goal.q_values}")

    return history


def main():
    print("QSE-ÉMILE EXPERIMENTS")
    print("=" * 30)

    # Run experiments
    hist1 = run_experiment("Mixed Rewards", 200, "mixed")
    hist2 = run_experiment("No Rewards", 200, "none")
    hist3 = run_experiment("Long Run", 400, "mixed")

    # Plot the long run
    print(f"\nPlotting Long Run results...")
    plot_surplus_sigma(hist3, dt=0.01)
    plot_context_timeline(hist3, dt=0.01)
    plot_goal_timeline(hist3, dt=0.01)

    print(f"\n✅ Done")


if __name__ == "__main__":
    main()
