"""
comprehensive_runner.py - Research-grade experimental runner for QSE-Émile

Runs extensive parameter sweeps, multiple trials, temporal analysis, and saves all data.
Designed for academic research and MRP thesis work.
FIXED VERSION: Efficient storage to prevent JSON corruption.
"""

import json
import numpy as np
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from config import CONFIG
from agent import EmileAgent
from viz import plot_surplus_sigma, plot_context_timeline, plot_goal_timeline


class ComprehensiveRunner:
    """Research-grade experimental runner with comprehensive analysis"""

    def __init__(self, save_dir="qse_emile_data"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def run_parameter_sweep(self):
        """Run systematic parameter sweep to explore the behavior space"""

        print("=== COMPREHENSIVE PARAMETER SWEEP ===")

        # Parameters to sweep
        param_sets = [
            # Baseline
            {"name": "baseline", "changes": {}},

            # Curvature coupling variations
            {"name": "low_curvature", "changes": {"S_BETA": 0.3}},
            {"name": "high_curvature", "changes": {"S_BETA": 0.8}},

            # Quantum coupling variations
            {"name": "low_quantum", "changes": {"QUANTUM_COUPLING": 0.05}},
            {"name": "high_quantum", "changes": {"QUANTUM_COUPLING": 0.25}},

            # Context sensitivity variations
            {"name": "sensitive_context", "changes": {"RECONTEXTUALIZATION_THRESHOLD": 0.1}},
            {"name": "stable_context", "changes": {"RECONTEXTUALIZATION_THRESHOLD": 0.4}},

            # Emergent time variations
            {"name": "fast_time", "changes": {"TAU_MIN": 0.05, "TAU_MAX": 0.5}},
            {"name": "slow_time", "changes": {"TAU_MIN": 0.3, "TAU_MAX": 2.0}},
        ]

        all_results = {}

        for param_set in param_sets:
            print(f"\n--- Testing {param_set['name']} ---")

            # Modify config
            original_values = {}
            for param, value in param_set['changes'].items():
                original_values[param] = getattr(CONFIG, param)
                setattr(CONFIG, param, value)

            # Run multiple trials
            trials = self.run_multiple_trials(
                name=param_set['name'],
                n_trials=3,
                steps=400
            )

            all_results[param_set['name']] = trials

            # Restore original values
            for param, value in original_values.items():
                setattr(CONFIG, param, value)

        return all_results

    def run_multiple_trials(self, name, n_trials=5, steps=300, reward_patterns=None):
        """Run multiple trials for statistical reliability"""

        if reward_patterns is None:
            reward_patterns = ["mixed", "periodic", "none"]

        all_trials = {}

        for pattern in reward_patterns:
            pattern_trials = []

            for trial in range(n_trials):
                print(f"  {name} - {pattern} - trial {trial+1}/{n_trials}")

                result = self.run_single_experiment(
                    name=f"{name}_{pattern}_trial{trial}",
                    steps=steps,
                    reward_pattern=pattern,
                    goals=["explore", "exploit", "maintain", "adapt", "consolidate"],
                    verbose=False
                )

                pattern_trials.append(result)

            all_trials[pattern] = pattern_trials

        return all_trials

    def run_single_experiment(self, name, steps, reward_pattern="mixed", goals=None, verbose=True):
        """Run a single comprehensive experiment with EFFICIENT storage"""

        if goals is None:
            goals = ["explore", "exploit", "maintain", "adapt", "consolidate"]

        if verbose:
            print(f"\n=== {name} ===")

        # Create agent
        agent = EmileAgent(CONFIG)
        for goal_id in goals:
            agent.goal.add_goal(goal_id)

        # Create reward schedule
        rewards = self._create_reward_schedule(steps, reward_pattern)

        # Track key metrics during simulation (EFFICIENT - not every step)
        surplus_samples = []
        sigma_samples = []
        context_changes = []
        goal_switches = []
        reward_responses = []

        # Sample every N steps instead of every step to prevent massive JSON
        sample_interval = max(1, steps // 50)  # Take ~50 samples max

        for t in range(steps):
            ext_input = {'reward': rewards[t]} if rewards[t] != 0.0 else None

            # Get pre-step state for event tracking
            pre_context = agent.context.get_current()
            pre_goal = agent.goal.current_goal

            # Step
            metrics = agent.step(dt=0.01, external_input=ext_input)

            # Sample key metrics periodically, not every step
            if t % sample_interval == 0:
                surplus_samples.append({
                    'step': t,
                    'surplus_mean': float(np.mean(agent.qse.S)),
                    'sigma_mean': float(metrics.get('sigma_mean', 0))
                })

            # Track significant events only
            if agent.context.get_current() != pre_context:
                context_changes.append({
                    'step': t,
                    'from': pre_context,
                    'to': agent.context.get_current()
                })

            if agent.goal.current_goal != pre_goal:
                goal_switches.append({
                    'step': t,
                    'from': pre_goal,
                    'to': agent.goal.current_goal
                })

            if rewards[t] != 0.0:
                reward_responses.append({
                    'step': t,
                    'reward': float(rewards[t]),
                    'goal': agent.goal.current_goal,
                    'surplus': float(np.mean(agent.qse.S))
                })

        # Get final results
        history = agent.get_history()

        # Comprehensive analysis
        analysis = self._analyze_experiment_summary(history, surplus_samples, rewards)

        if verbose:
            self._print_analysis(analysis)

        # Return SUMMARY data only - no massive arrays
        return {
            'name': name,
            'config_used': self._serialize_config(),
            'summary_stats': {
                'total_steps': steps,
                'total_rewards': float(np.sum(np.abs(rewards))),
                'final_surplus_mean': float(np.mean(agent.qse.S)),
                'context_changes': len(context_changes),
                'goal_switches': len(goal_switches)
            },
            'analysis': analysis,
            'final_q_values': {k: float(v) for k, v in agent.goal.q_values.items()},
            'goals': goals,
            'reward_pattern': reward_pattern,
            'key_events': {
                'context_changes': context_changes[:10],  # First 10 only
                'goal_switches': goal_switches[:10],      # First 10 only
                'reward_responses': reward_responses[:10] # First 10 only
            },
            'time_series_summary': {
                'surplus_samples': surplus_samples,  # ~50 samples instead of 400+
                'n_samples': len(surplus_samples),
                'sample_interval': sample_interval
            }
        }

    def _create_reward_schedule(self, steps, pattern):
        """Create reward schedule based on pattern"""
        rewards = np.zeros(steps)

        if pattern == "mixed":
            for i in range(steps):
                if i % 25 == 0:
                    rewards[i] = 0.8
                elif i % 40 == 0:
                    rewards[i] = 0.6
                elif i % 71 == 0:
                    rewards[i] = 1.0
                elif np.random.random() < 0.03:
                    rewards[i] = np.random.uniform(0.3, 0.7)
                elif np.random.random() < 0.01:
                    rewards[i] = -0.2

        elif pattern == "periodic":
            for i in range(0, steps, 20):
                rewards[i] = 0.8

        elif pattern == "sparse":
            for i in range(0, steps, 50):
                rewards[i] = 1.0

        elif pattern == "random":
            for i in range(steps):
                if np.random.random() < 0.05:
                    rewards[i] = np.random.uniform(0.3, 0.9)
                elif np.random.random() < 0.02:
                    rewards[i] = -0.3

        # "none" pattern keeps all zeros

        return rewards

    def _analyze_experiment_summary(self, history, surplus_samples, rewards):
        """Analyze experiment with summary data only"""

        surplus = np.array(history['surplus_mean'])
        sigma = np.array(history['sigma_mean'])
        contexts = np.array(history['context_id'])
        goals = history['goal']

        analysis = {}

        # Basic statistics
        analysis['surplus_stats'] = {
            'mean': float(surplus.mean()),
            'std': float(surplus.std()),
            'min': float(surplus.min()),
            'max': float(surplus.max())
        }

        analysis['sigma_stats'] = {
            'mean': float(sigma.mean()),
            'std': float(sigma.std()),
            'min': float(sigma.min()),
            'max': float(sigma.max())
        }

        # Context analysis
        analysis['context_stats'] = {
            'total_shifts': len(set(contexts)),
            'final_context': int(contexts[-1]),
            'shift_frequency': float(len(set(contexts)) / len(contexts))
        }

        # Goal analysis
        active_goals = [g for g in goals if g is not None]
        if active_goals:
            goal_counts = Counter(active_goals)
            analysis['goal_stats'] = {
                'distribution': dict(goal_counts),
                'diversity': len(set(active_goals)),
                'most_frequent': goal_counts.most_common(1)[0] if goal_counts else None
            }
        else:
            analysis['goal_stats'] = {'distribution': {}, 'diversity': 0}

        # Oscillation analysis (using efficient samples)
        if len(surplus_samples) > 10:
            surplus_ts = [s['surplus_mean'] for s in surplus_samples]
            sigma_ts = [s['sigma_mean'] for s in surplus_samples]

            # FFT analysis on samples
            if len(surplus_ts) > 4:
                from scipy.fft import fft
                sigma_fft = np.abs(fft(np.array(sigma_ts) - np.mean(sigma_ts)))
                if len(sigma_fft) > 2:
                    dom_freq_idx = np.argmax(sigma_fft[1:len(sigma_fft)//2]) + 1
                    sigma_period = len(sigma_ts) / dom_freq_idx
                else:
                    sigma_period = None
            else:
                sigma_period = None

            analysis['oscillation_summary'] = {
                'surplus_sigma_correlation': float(np.corrcoef(surplus_ts, sigma_ts)[0,1]),
                'surplus_range': float(max(surplus_ts) - min(surplus_ts)),
                'sigma_range': float(max(sigma_ts) - min(sigma_ts)),
                'estimated_period': float(sigma_period) if sigma_period else None
            }

        # Reward analysis
        reward_steps = [i for i, r in enumerate(rewards) if r != 0]
        analysis['reward_summary'] = {
            'total_reward_events': len(reward_steps),
            'total_reward_magnitude': float(np.sum(np.abs(rewards)))
        }

        return analysis

    def _serialize_config(self):
        """Serialize CONFIG for saving"""
        config_dict = {}
        for attr in dir(CONFIG):
            if not attr.startswith('_'):
                value = getattr(CONFIG, attr)
                if isinstance(value, (int, float, str, bool)):
                    config_dict[attr] = value
        return config_dict

    def _print_analysis(self, analysis):
        """Print analysis results"""
        print(f"Surplus: {analysis['surplus_stats']['mean']:.3f} ± {analysis['surplus_stats']['std']:.3f}")
        print(f"Sigma: {analysis['sigma_stats']['mean']:.3f} ± {analysis['sigma_stats']['std']:.3f}")
        print(f"Context shifts: {analysis['context_stats']['total_shifts']}")

        if analysis['goal_stats']['distribution']:
            print(f"Goal distribution: {analysis['goal_stats']['distribution']}")

        if 'oscillation_summary' in analysis and analysis['oscillation_summary']['estimated_period']:
            print(f"Estimated period: {analysis['oscillation_summary']['estimated_period']:.1f} samples")

    def save_results(self, results, filename=None):
        """Save SUMMARY results only - much smaller JSON"""

        if filename is None:
            filename = f"summary_results_{self.experiment_id}.json"

        filepath = self.save_dir / filename

        # Convert numpy types
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            else:
                return obj

        # Create summary version of results instead of full data dump
        summary_results = {
            'experiment_id': results.get('experiment_id'),
            'timestamp': results.get('timestamp'),
            'config_baseline': results.get('config_baseline'),
            'summary_stats': {
                'total_experiments': len(results.get('parameter_sweep', {})) + len(results.get('long_evolution', {})),
                'parameter_conditions': list(results.get('parameter_sweep', {}).keys()) if 'parameter_sweep' in results else [],
                'evolution_lengths': list(results.get('long_evolution', {}).keys()) if 'long_evolution' in results else []
            },
            'parameter_sweep_summary': {},
            'long_evolution_summary': {},
            'goal_set_summary': {}
        }

        # Summarize parameter sweep (key findings only)
        if 'parameter_sweep' in results:
            for param_name, trials in results['parameter_sweep'].items():
                if 'mixed' in trials and trials['mixed']:
                    trial = trials['mixed'][0]  # First trial
                    summary_results['parameter_sweep_summary'][param_name] = {
                        'surplus_mean': trial['analysis']['surplus_stats']['mean'],
                        'sigma_mean': trial['analysis']['sigma_stats']['mean'],
                        'context_shifts': trial['analysis']['context_stats']['total_shifts'],
                        'final_q_values': trial['final_q_values']
                    }

        # Summarize long evolution
        if 'long_evolution' in results:
            for length_key, data in results['long_evolution'].items():
                summary_results['long_evolution_summary'][length_key] = {
                    'surplus_mean': data['analysis']['surplus_stats']['mean'],
                    'sigma_mean': data['analysis']['sigma_stats']['mean'],
                    'context_shifts': data['analysis']['context_stats']['total_shifts'],
                    'final_q_values': data['final_q_values']
                }

        # Summarize goal variations
        if 'goal_set_variations' in results:
            for var_name, data in results['goal_set_variations'].items():
                summary_results['goal_set_summary'][var_name] = {
                    'n_goals': len(data['goals']),
                    'surplus_mean': data['analysis']['surplus_stats']['mean'],
                    'context_shifts': data['analysis']['context_stats']['total_shifts'],
                    'final_q_values': data['final_q_values']
                }

        results_serializable = convert_numpy(summary_results)

        # Save smaller JSON
        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)

        print(f"\nSUMMARY results saved to: {filepath}")
        print(f"File size: {filepath.stat().st_size / 1024:.1f} KB (instead of MB)")
        return filepath

    def run_comprehensive_suite(self):
        """Run the full comprehensive experimental suite"""

        print("QSE-ÉMILE COMPREHENSIVE RESEARCH SUITE")
        print("=" * 50)
        print(f"Experiment ID: {self.experiment_id}")

        start_time = time.time()

        # 1. Parameter sweep
        print("\n1. Running parameter sweep...")
        param_results = self.run_parameter_sweep()

        # 2. Long evolution studies
        print("\n2. Running long evolution studies...")
        long_results = {}
        for steps in [500, 1000, 1500]:
            print(f"  {steps} step evolution...")
            result = self.run_single_experiment(
                name=f"evolution_{steps}",
                steps=steps,
                reward_pattern="mixed",
                verbose=False
            )
            long_results[f"steps_{steps}"] = result

        # 3. Goal set variations
        print("\n3. Testing different goal sets...")
        goal_set_results = {}

        goal_sets = [
            (["explore", "exploit"], "basic"),
            (["explore", "exploit", "maintain"], "moderate"),
            (["explore", "exploit", "maintain", "adapt", "consolidate"], "full"),
            (["explore", "exploit", "maintain", "adapt", "consolidate", "create", "destroy"], "extended")
        ]

        for goals, name in goal_sets:
            result = self.run_single_experiment(
                name=f"goalset_{name}",
                steps=300,
                goals=goals,
                verbose=False
            )
            goal_set_results[name] = result

        # Compile all results
        comprehensive_results = {
            'experiment_id': self.experiment_id,
            'timestamp': datetime.now().isoformat(),
            'parameter_sweep': param_results,
            'long_evolution': long_results,
            'goal_set_variations': goal_set_results,
            'config_baseline': self._serialize_config()
        }

        # Save results
        filepath = self.save_results(comprehensive_results)

        end_time = time.time()
        print(f"\n✅ COMPREHENSIVE SUITE COMPLETE")
        print(f"Total time: {end_time - start_time:.1f} seconds")
        print(f"Results saved to: {filepath}")

        return comprehensive_results


def main():
    """Run the comprehensive research suite"""
    runner = ComprehensiveRunner()
    results = runner.run_comprehensive_suite()
    return runner, results


if __name__ == "__main__":
    runner, results = main()
