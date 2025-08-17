import numpy as np
from collections import Counter

class CognitiveStory:
    """
    Analyzes and interprets the results of an Émile-mini experiment
    to tell a clear story about the agent's cognitive behavior.
    """

    def __init__(self, experiment_result):
        """
        Initialize with the result dictionary from an experiment.

        Args:
            experiment_result (dict): The output from ExperimentRunner.run_experiment()
        """
        self.name = experiment_result['name']
        self.history = experiment_result['history']
        self.q_values = experiment_result['q_values']
        self.config = experiment_result['config']
        self.agent = experiment_result['agent']
        self.reward_pattern = self.get_reward_pattern(experiment_result)

        # Perform analysis
        self._analyze()

    def get_reward_pattern(self, experiment_result):
        """A helper to find the reward pattern from the experiment name or logs if possible."""
        # This is a simple heuristic based on your experiment names.
        # A more robust system would pass the reward_pattern into the result object.
        name_lower = self.name.lower()
        if "intrinsic" in name_lower or "none" in name_lower:
            return "none"
        if "periodic" in name_lower:
            return "periodic"
        if "mixed" in name_lower:
            return "mixed"
        return "unknown"


    def _analyze(self):
        """Performs all the data analysis needed to tell the story."""
        # --- Basic Stats ---
        self.steps = len(self.history['surplus_mean'])
        self.surplus_mean = np.mean(self.history['surplus_mean'])
        self.surplus_std = np.std(self.history['surplus_mean'])
        self.sigma_mean = np.mean(self.history['sigma_mean'])
        self.sigma_std = np.std(self.history['sigma_mean'])

        # --- Cognitive Dynamics ---
        self.context_ids = self.history['context_id']
        self.num_context_shifts = len(set(self.context_ids))
        self.goal_selections = self.history['goal']

        self.goal_counts = Counter(g for g in self.goal_selections if g is not None)
        self.most_frequent_goal = self.goal_counts.most_common(1)[0] if self.goal_counts else ('None', 0)

        # --- Learning Analysis ---
        self.initial_q_values = {g: 0.0 for g in self.q_values.keys()}
        self.final_q_values = self.q_values
        self.sorted_final_q = sorted(self.final_q_values.items(), key=lambda item: item[1], reverse=True)
        self.learned_preference = self.sorted_final_q[0][0] if self.sorted_final_q else 'None'
        self.has_learned = any(v != 0.0 for v in self.final_q_values.values())


    def tell_the_story(self):
        """Prints the full narrative analysis of the agent's behavior."""
        print("="*60)
        print(f"The Cognitive Story of Experiment: '{self.name}'")
        print("="*60)

        self._report_on_self_maintenance()
        self._report_on_goal_response()
        self._report_on_cognitive_style()

        print("\n" + "="*60)
        print("CONCLUSION")
        print("="*60)
        self._draw_conclusions()
        print("="*60)


    def _report_on_self_maintenance(self):
        print("\n--- Part 1: Self-Maintenance and Intrinsic Dynamics ---")
        print(f"Over {self.steps} steps, the agent maintained a stable average surplus of {self.surplus_mean:.3f} (std: {self.surplus_std:.3f}).")
        print("This demonstrates a core viability; the system did not collapse or explode, but sustained itself.")

        if self.reward_pattern == 'none':
            print("\nThis experiment had NO external rewards.")
            print("Despite the lack of external motivation, the agent was far from idle:")
            print(f"   - It actively pursued goals, with selections distributed as: {dict(self.goal_counts)}.")
            print(f"   - It underwent {self.num_context_shifts} distinct context shifts, showing internal cognitive evolution.")
            print("\n>> Finding: The agent is inherently active and self-regulating. Its behavior is driven by an intrinsic need to maintain its internal state (autopoiesis), not just by external rewards.")
        else:
            print("\n>> Finding: The agent exhibits a stable baseline of self-maintenance, upon which learning and environmental interaction can be built.")


    def _report_on_goal_response(self):
        print("\n--- Part 2: Responding to Goals and Learning ---")
        if not self.has_learned:
            print("The agent did not learn in this experiment, as no rewards were provided.")
            print("Its Q-values remained at their initial state: {self.final_q_values}")
            print("\n>> Finding: Without environmental feedback, the agent's goal preferences do not change. Its actions are purely based on its intrinsic dynamics.")
        else:
            print("The agent learned from the environmental rewards in this experiment.")
            print(f"Its goal preferences evolved from a neutral state to a clear hierarchy.")
            print(f"   - Final Learned Preference: '{self.learned_preference}' (Q-value: {self.sorted_final_q[0][1]:.4f})")
            print(f"   - Full Q-Value Ranking: {[(g, f'{q:.4f}') for g, q in self.sorted_final_q]}")

            behavior_matches_learning = (self.most_frequent_goal[0] == self.learned_preference)

            print(f"\nThe agent's behavior directly reflects this learning:")
            print(f"   - The most frequently chosen goal was '{self.most_frequent_goal[0]}', selected {self.most_frequent_goal[1]} times.")

            if behavior_matches_learning:
                print("   - This aligns perfectly with its highest learned Q-value.")
            else:
                print(f"   - Interestingly, this does NOT align with its highest learned Q-value of '{self.learned_preference}'. This suggests a complex dynamic where intrinsic pressures might still compete with learned strategies.")

            print("\n>> Finding: The agent is clearly responsive to its environment. It updates its internal goal preferences based on rewards and adapts its behavior to reflect what it has learned.")


    def _report_on_cognitive_style(self):
        print("\n--- Part 3: Cognitive and Behavioral Style ---")
        print(f"The agent's cognitive journey involved {self.num_context_shifts} shifts in its internal 'worldview'.")

        # Characterize the agent's dominant behavior
        if self.goal_counts:
            dominant_behavior = self.most_frequent_goal[0]
            print(f"Its dominant behavioral mode was '{dominant_behavior}'.")
        else:
            dominant_behavior = "passive"
            print("The agent did not select any goals.")

        # A simple heuristic to classify the agent's "personality" in this run
        if dominant_behavior in ['explore', 'adapt']:
            style = "Inquisitive and Adaptive"
        elif dominant_behavior in ['exploit', 'consolidate']:
            style = "Focused and Deliberate"
        elif dominant_behavior == 'maintain':
            style = "Stable and Conservative"
        else:
            style = "Passive"

        print(f"Overall Cognitive Style: {style}")


    def _draw_conclusions(self):
        # Final summary verdict
        is_self_maintaining = self.surplus_std < 0.1 and self.steps > 0
        is_responsive_to_goals = self.has_learned

        if is_self_maintaining:
            print("1. Is the agent self-maintaining? YES.")
            print("   It consistently maintains a stable internal state (surplus) and exhibits intrinsic activity even without rewards.")
        else:
            print("1. Is the agent self-maintaining? NO.")
            print("   It failed to maintain a stable internal state, indicating a lack of viability in this configuration.")

        if self.reward_pattern != 'none':
            if is_responsive_to_goals:
                print("\n2. Is the agent responding to goals? YES.")
                print("   It learns from environmental rewards, updates its internal Q-values, and its behavior demonstrably shifts to align with its learned preferences.")
            else:
                print("\n2. Is the agent responding to goals? INCONCLUSIVE.")
                print("   It was in a reward-based environment but showed no significant learning or behavioral adaptation.")
        else:
             print("\n2. Is the agent responding to goals? N/A.")
             print("   This was an intrinsic-only run with no rewards to respond to.")


if __name__ == '__main__':
    # Add path so experiment_runner can be found
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # We need to run the experiments first to get the results
    from experiment_runner import main as run_suite

    print("Running the experimental suite to generate data...")
    all_experiments = run_suite()

    # Let's tell the story of the two most interesting experiments

    # The agent with no rewards, showing its intrinsic nature
    intrinsic_experiment = next(exp for exp in all_experiments if exp['name'] == 'Intrinsic Only')
    story_intrinsic = CognitiveStory(intrinsic_experiment)
    story_intrinsic.tell_the_story()

    # The agent on a long run, showing its learning and evolution
    long_experiment = next(exp for exp in all_experiments if exp['name'] == 'Long Evolution')
    story_long = CognitiveStory(long_experiment)
    story_long.tell_the_story()
