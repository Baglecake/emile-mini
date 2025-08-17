"""
extinction_experiment.py - Test agent resilience during reward extinction

Tests whether QSE-Émile's intrinsic dynamics help maintain learned knowledge
better than standard RL during periods without external rewards.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from agent import EmileAgent
from config import CONFIG


class StandardRLAgent:
    """Standard Q-learning agent for comparison"""

    def __init__(self, goals, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.goals = goals
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_values = {goal: 0.0 for goal in goals}
        self.current_goal = None
        self.history = {
            'goal': [],
            'q_values': [],
            'total_reward': 0.0
        }

    def select_goal(self):
        """Select a goal using epsilon-greedy strategy"""
        if np.random.random() < self.epsilon:
            self.current_goal = np.random.choice(self.goals)
        else:
            # Exploit: choose goal with highest Q-value
            best_goals = [g for g in self.goals if self.q_values[g] == max(self.q_values.values())]
            self.current_goal = np.random.choice(best_goals)
        return self.current_goal

    def update_q_value(self, goal, reward):
        """Update Q-value using simple temporal difference learning"""
        if goal in self.q_values:
            old_value = self.q_values[goal]
            # Simple Q-update without next state (episodic)
            self.q_values[goal] = old_value + self.learning_rate * (reward - old_value)

    def step(self, reward=0.0):
        """Perform one step"""
        goal = self.select_goal()
        self.update_q_value(goal, reward)

        # Record history
        self.history['goal'].append(goal)
        self.history['q_values'].append(dict(self.q_values))
        self.history['total_reward'] += reward

        return goal


def create_extinction_schedule(phase1_steps, phase2_steps, phase3_steps,
                             reward_value=1.0, period=10):
    """Create three-phase reward schedule: Learning -> Extinction -> Recovery"""

    total_steps = phase1_steps + phase2_steps + phase3_steps
    rewards = np.zeros(total_steps)

    # Phase 1: Learning period - periodic rewards
    for i in range(period, phase1_steps, period):
        if i < phase1_steps:
            rewards[i] = reward_value

    # Phase 2: Extinction - no rewards (already zeros)

    # Phase 3: Recovery - rewards return
    recovery_start = phase1_steps + phase2_steps
    for i in range(recovery_start + period, total_steps, period):
        rewards[i] = reward_value

    return rewards


def run_extinction_experiment(phase1=150, phase2=200, phase3=150, n_trials=3):
    """Run the extinction resilience experiment"""

    print("🧪 EXTINCTION RESILIENCE EXPERIMENT")
    print("=" * 50)
    print("Testing knowledge preservation during reward extinction")
    print(f"Phase 1 (Learning): {phase1} steps")
    print(f"Phase 2 (Extinction): {phase2} steps")
    print(f"Phase 3 (Recovery): {phase3} steps")

    goals = ["explore", "exploit", "maintain", "adapt"]
    total_steps = phase1 + phase2 + phase3

    # Storage for multiple trials
    emile_trials = []
    standard_trials = []

    for trial in range(n_trials):
        print(f"\nRunning trial {trial + 1}/{n_trials}...")

        # Create reward schedule
        rewards = create_extinction_schedule(phase1, phase2, phase3)

        # --- QSE-Émile Agent ---
        emile_agent = EmileAgent(CONFIG)
        for goal in goals:
            emile_agent.goal.add_goal(goal)

        emile_q_history = []
        emile_context_history = []
        emile_surplus_history = []

        for t in range(total_steps):
            reward = rewards[t]
            external_input = {'reward': reward} if reward > 0 else None

            # Step the agent
            metrics = emile_agent.step(dt=0.01, external_input=external_input)

            # Record state
            emile_q_history.append(dict(emile_agent.goal.q_values))
            emile_context_history.append(emile_agent.context.get_current())
            emile_surplus_history.append(float(np.mean(emile_agent.qse.S)))

        # --- Standard RL Agent ---
        standard_agent = StandardRLAgent(goals)

        for t in range(total_steps):
            reward = rewards[t]
            standard_agent.step(reward)

        # Store trial results
        emile_trials.append({
            'q_values': emile_q_history,
            'contexts': emile_context_history,
            'surplus': emile_surplus_history,
            'final_q': dict(emile_agent.goal.q_values)
        })

        standard_trials.append({
            'q_values': standard_agent.history['q_values'],
            'final_q': dict(standard_agent.q_values)
        })

    # Analyze and visualize results
    analyze_extinction_results(emile_trials, standard_trials, phase1, phase2, phase3, rewards)

    return emile_trials, standard_trials


def analyze_extinction_results(emile_trials, standard_trials, phase1, phase2, phase3, rewards):
    """Analyze and visualize extinction experiment results"""

    total_steps = phase1 + phase2 + phase3
    time_axis = np.arange(total_steps)

    # Extract exploit Q-values for primary analysis
    emile_exploit_q = []
    standard_exploit_q = []

    for trial in emile_trials:
        exploit_q = [q_vals.get('exploit', 0.0) for q_vals in trial['q_values']]
        emile_exploit_q.append(exploit_q)

    for trial in standard_trials:
        exploit_q = [q_vals.get('exploit', 0.0) for q_vals in trial['q_values']]
        standard_exploit_q.append(exploit_q)

    # Calculate means and std for plotting
    emile_mean = np.mean(emile_exploit_q, axis=0)
    emile_std = np.std(emile_exploit_q, axis=0)
    standard_mean = np.mean(standard_exploit_q, axis=0)
    standard_std = np.std(standard_exploit_q, axis=0)

    # Create visualization
    plt.figure(figsize=(14, 8))

    # Plot Q-value evolution
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, standard_mean, 'r--', label='Standard RL', linewidth=2)
    plt.fill_between(time_axis, standard_mean - standard_std, standard_mean + standard_std,
                     color='red', alpha=0.2)

    plt.plot(time_axis, emile_mean, 'b-', label='QSE-Émile', linewidth=2)
    plt.fill_between(time_axis, emile_mean - emile_std, emile_mean + emile_std,
                     color='blue', alpha=0.2)

    # Add phase markers
    plt.axvline(x=phase1, color='orange', linestyle=':', linewidth=2, label='Extinction Begins')
    plt.axvline(x=phase1 + phase2, color='green', linestyle=':', linewidth=2, label='Recovery Begins')

    # Add phase backgrounds
    plt.axvspan(0, phase1, alpha=0.1, color='green', label='Learning Phase')
    plt.axvspan(phase1, phase1 + phase2, alpha=0.1, color='gray', label='Extinction Phase')
    plt.axvspan(phase1 + phase2, total_steps, alpha=0.1, color='lightblue', label='Recovery Phase')

    plt.title("Agent Resilience During Reward Extinction", fontsize=16, fontweight='bold')
    plt.ylabel("Q-Value for 'exploit' Goal", fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)

    # Plot reward schedule
    plt.subplot(2, 1, 2)
    plt.plot(time_axis, rewards, 'k-', linewidth=1, alpha=0.7)
    plt.fill_between(time_axis, 0, rewards, alpha=0.3, color='gold')
    plt.ylabel("Reward", fontsize=12)
    plt.xlabel("Time Steps", fontsize=12)
    plt.title("Reward Schedule", fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("extinction_resilience_experiment.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Quantitative analysis
    print("\n" + "=" * 50)
    print("EXTINCTION RESILIENCE ANALYSIS")
    print("=" * 50)

    # Q-value preservation during extinction
    extinction_start = phase1
    extinction_end = phase1 + phase2
    recovery_end = total_steps

    # Calculate key metrics
    emile_pre_extinction = np.mean([trial[extinction_start-1] for trial in emile_exploit_q])
    emile_post_extinction = np.mean([trial[extinction_end-1] for trial in emile_exploit_q])
    emile_final_recovery = np.mean([trial[recovery_end-1] for trial in emile_exploit_q])

    standard_pre_extinction = np.mean([trial[extinction_start-1] for trial in standard_exploit_q])
    standard_post_extinction = np.mean([trial[extinction_end-1] for trial in standard_exploit_q])
    standard_final_recovery = np.mean([trial[recovery_end-1] for trial in standard_exploit_q])

    # Calculate preservation rates
    emile_preservation = (emile_post_extinction / emile_pre_extinction) if emile_pre_extinction > 0 else 0
    standard_preservation = (standard_post_extinction / standard_pre_extinction) if standard_pre_extinction > 0 else 0

    # Calculate recovery rates
    emile_recovery = (emile_final_recovery / emile_pre_extinction) if emile_pre_extinction > 0 else 0
    standard_recovery = (standard_final_recovery / standard_pre_extinction) if standard_pre_extinction > 0 else 0

    print(f"KNOWLEDGE PRESERVATION DURING EXTINCTION:")
    print(f"  QSE-Émile:   {emile_preservation:.2%} of pre-extinction knowledge preserved")
    print(f"  Standard RL: {standard_preservation:.2%} of pre-extinction knowledge preserved")
    print(f"  → Advantage: {(emile_preservation - standard_preservation):.2%}")

    print(f"\nRECOVERY AFTER EXTINCTION:")
    print(f"  QSE-Émile:   {emile_recovery:.2%} recovery vs pre-extinction")
    print(f"  Standard RL: {standard_recovery:.2%} recovery vs pre-extinction")
    print(f"  → Advantage: {(emile_recovery - standard_recovery):.2%}")

    print(f"\nFINAL Q-VALUES:")
    print(f"  QSE-Émile:   {emile_final_recovery:.4f}")
    print(f"  Standard RL: {standard_final_recovery:.4f}")

    # Determine conclusion
    if emile_preservation > standard_preservation + 0.1:
        print(f"\n🎯 KEY FINDINGS:")
        print(f"✅ QSE-Émile shows superior knowledge preservation during extinction")
        print(f"✅ Intrinsic dynamics maintain learned values without external rewards")
        if emile_recovery > standard_recovery + 0.1:
            print(f"✅ Faster and more complete recovery when rewards return")
        print(f"\n→ CONCLUSION: Strong evidence for intrinsic knowledge maintenance")
    else:
        print(f"\n→ CONCLUSION: Mixed results - may need parameter tuning")

    return {
        'emile_preservation': emile_preservation,
        'standard_preservation': standard_preservation,
        'emile_recovery': emile_recovery,
        'standard_recovery': standard_recovery
    }


def main():
    """Run the extinction resilience demonstration"""
    print("🧬 QSE-ÉMILE: KNOWLEDGE RESILIENCE DEMONSTRATOR")
    print("=" * 60)
    print("Testing intrinsic knowledge preservation during reward extinction")

    results = run_extinction_experiment(phase1=150, phase2=200, phase3=150, n_trials=3)

    print(f"\n🏆 DEMONSTRATION COMPLETE")
    print("Key Innovation: Intrinsic dynamics preserve knowledge without external rewards")
    print("Complementary to: Context-switching for escaping local optima")


if __name__ == "__main__":
    main()
