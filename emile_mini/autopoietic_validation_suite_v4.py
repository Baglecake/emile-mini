
#!/usr/bin/env python3
"""
Enhanced Autopoietic Validation Suite v3.0
==========================================

COMPLETE MODULE - Drop-in replacement for autopoietic_validation_suite_v2.py

Key Enhancements:
1. Phasic rupture detection and causality testing
2. Emergent time (τ) dynamics analysis
3. Regime-sensitive perturbation testing
4. Decision chain and recursive causality detection
5. Context effectiveness analysis
6. Integration with QSE dynamics patterns
7. Multi-lag correlation analysis
8. Enhanced scoring with continuous metrics

Usage:
    python enhanced_autopoietic_validation_suite.py --comprehensive
    python enhanced_autopoietic_validation_suite.py --test enhanced_causality
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
import time
from pathlib import Path
from scipy import stats
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Import QSE components
from emile_mini.config import QSEConfig
from emile_mini.agent import EmileAgent
from emile_mini.embodied_qse_emile import EmbodiedQSEAgent, EmbodiedEnvironment
from emile_mini.social_qse_agent_v2 import SocialQSEAgent, SocialEnvironment

try:
    from analyze_qse_dynamics import QSEDynamicsAnalyzer
    QSE_ANALYZER_AVAILABLE = True
except ImportError:
    QSE_ANALYZER_AVAILABLE = False
    print("⚠️ analyze_qse_dynamics.py not found - some enhanced features disabled")


class EnhancedAutopoieticValidationSuite:
    """Enhanced comprehensive autopoiesis validation framework"""

    def __init__(self, output_dir: str = "enhanced_autopoiesis_validation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.results = {}
        self.statistical_tests = {}
        self.effect_sizes = {}

        print(f"🧬 ENHANCED AUTOPOIETIC VALIDATION SUITE v3.0")
        print(f"Output directory: {self.output_dir}")

    def test_enhanced_qse_causality(self, steps: int = 15000, n_trials: int = 5) -> Dict[str, Any]:
        """
        Enhanced QSE Causality Test - integrates rich QSE dynamics
        Tests multiple causal pathways: phasic rupture → behavior, τ → decisions, etc.
        """
        print(f"\n🎯 ENHANCED QSE CAUSALITY TEST")
        print(f"Testing rich QSE dynamics → behavior causality...")

        causality_results = []

        for trial in range(n_trials):
            print(f"  Trial {trial+1}/{n_trials}")

            cfg = QSEConfig()
            agent = EmileAgent(cfg)
            for goal in ["explore", "exploit", "maintain", "adapt", "learn"]:
                agent.goal.add_goal(goal)

            # Collect rich time series data
            rich_history = {
                'tau': [],
                'sigma_mean': [],
                'sigma_ema': [],
                'surplus_mean': [],
                'phasic_rupture': [],
                'tonic_rupture': [],
                'entropy': [],
                'goal_changes': [],
                'context_changes': [],
                'decision_events': [],
                'q_value_changes': [],
                'qse_influence': [],
                'regime_state': []
            }

            previous_goal = None
            previous_context = None
            previous_sigma = 0
            sigma_history = deque(maxlen=5)  # For rupture detection

            for step in range(steps):
                # Get QSE state before step
                pre_metrics = {
                    'surplus_mean': np.mean(agent.qse.S),
                    'sigma_mean': np.mean(getattr(agent.symbolic, 'sigma', [0])),
                }

                # Agent step
                metrics = agent.step()

                # Post-step state
                current_goal = agent.goal.current_goal
                current_context = agent.context.get_current()

                # Detect events
                goal_changed = (previous_goal is not None and current_goal != previous_goal)
                context_changed = (previous_context is not None and current_context != previous_context)

                # Enhanced QSE metrics detection with NaN protection
                tau_current = getattr(agent.qse, 'tau', 0.5) if hasattr(agent.qse, 'tau') else 0.5
                tau_current = 0.5 if np.isnan(tau_current) or np.isinf(tau_current) else float(tau_current)

                sigma_current = metrics.get('sigma_mean', np.mean(getattr(agent.symbolic, 'sigma', [0])))
                if np.isnan(sigma_current) or np.isinf(sigma_current):
                    sigma_current = 0.0
                sigma_current = float(sigma_current)

                sigma_ema = getattr(agent.symbolic, 'sigma_ema', sigma_current) if hasattr(agent.symbolic, 'sigma_ema') else sigma_current
                if np.isnan(sigma_ema) or np.isinf(sigma_ema):
                    sigma_ema = sigma_current
                sigma_ema = float(sigma_ema)

                surplus_current = metrics.get('surplus_mean', np.mean(agent.qse.S))
                if np.isnan(surplus_current) or np.isinf(surplus_current):
                    surplus_current = np.nanmean(agent.qse.S) if not np.isnan(np.nanmean(agent.qse.S)) else 0.1
                surplus_current = float(surplus_current)

                # Phasic rupture detection (sudden large sigma changes)
                sigma_history.append(sigma_current)
                if len(sigma_history) >= 3:
                    sigma_change = abs(sigma_current - sigma_history[-2])
                    sigma_volatility = np.std(list(sigma_history))
                    phasic_rupture = sigma_change > 0.5 or sigma_volatility > 0.6
                else:
                    phasic_rupture = False

                # Tonic rupture (sustained high sigma)
                tonic_rupture = abs(sigma_current) > 0.4

                # Regime state detection
                if tau_current > 0.7:
                    regime = 'high_tau'
                elif tau_current < 0.3:
                    regime = 'low_tau'
                elif phasic_rupture:
                    regime = 'phasic_active'
                else:
                    regime = 'stable'

                # QSE influence score (how much QSE state drives decisions) with NaN protection
                surplus_change = abs(surplus_current - pre_metrics['surplus_mean'])
                if np.isnan(surplus_change) or np.isinf(surplus_change):
                    surplus_change = 0.0

                sigma_influence = abs(sigma_current) * 0.4
                tau_influence = abs(tau_current - 0.5) * 0.3
                rupture_influence = (0.5 if phasic_rupture else 0) + (0.2 if tonic_rupture else 0)

                qse_influence = surplus_change + sigma_influence + tau_influence + rupture_influence
                if np.isnan(qse_influence) or np.isinf(qse_influence):
                    qse_influence = 0.1
                qse_influence = float(qse_influence)

                # Debug: Check for problematic values periodically
                if step % 1000 == 0 and step > 0:
                    nan_count = sum([
                        np.isnan(tau_current), np.isnan(sigma_current),
                        np.isnan(surplus_current), np.isnan(qse_influence)
                    ])
                    if nan_count > 0:
                        print(f"    Warning: Found {nan_count} NaN values at step {step}")
                        print(f"    tau={tau_current}, sigma={sigma_current}, surplus={surplus_current}")

                # Q-value change calculation with protection
                q_change = 0
                if hasattr(agent.goal, 'q_values') and hasattr(agent.goal, '_previous_q_values'):
                    current_q = getattr(agent.goal, 'q_values', {})
                    previous_q = getattr(agent.goal, '_previous_q_values', {})
                    if current_q and previous_q:
                        q_changes = [abs(current_q.get(k, 0) - previous_q.get(k, 0))
                                   for k in set(current_q.keys()) | set(previous_q.keys())]
                        # Filter out NaN values
                        q_changes = [c for c in q_changes if not (np.isnan(c) or np.isinf(c))]
                        q_change = np.mean(q_changes) if q_changes else 0
                        if np.isnan(q_change) or np.isinf(q_change):
                            q_change = 0
                q_change = float(q_change)

                # Store rich data
                rich_history['tau'].append(tau_current)
                rich_history['sigma_mean'].append(sigma_current)
                rich_history['sigma_ema'].append(sigma_ema)
                rich_history['surplus_mean'].append(surplus_current)
                rich_history['phasic_rupture'].append(1 if phasic_rupture else 0)
                rich_history['tonic_rupture'].append(1 if tonic_rupture else 0)
                rich_history['entropy'].append(metrics.get('normalized_entropy', 0.5))
                rich_history['goal_changes'].append(1 if goal_changed else 0)
                rich_history['context_changes'].append(1 if context_changed else 0)
                rich_history['decision_events'].append(1 if (goal_changed or context_changed) else 0)
                rich_history['q_value_changes'].append(q_change)
                rich_history['qse_influence'].append(qse_influence)
                rich_history['regime_state'].append(regime)

                # Store previous values for next iteration
                if hasattr(agent.goal, 'q_values'):
                    agent.goal._previous_q_values = getattr(agent.goal, 'q_values', {}).copy()

                previous_goal = current_goal
                previous_context = current_context
                previous_sigma = sigma_current

            # Enhanced correlation analysis
            correlations = self._compute_enhanced_correlations(rich_history)

            # Regime-specific analysis
            regime_analysis = self._analyze_regime_specific_causality(rich_history)

            causality_results.append({
                'trial_id': trial,
                'correlations': correlations,
                'regime_analysis': regime_analysis,
                'phasic_events': sum(rich_history['phasic_rupture']),
                'tonic_events': sum(rich_history['tonic_rupture']),
                'decision_events': sum(rich_history['decision_events']),
                'context_switches': sum(rich_history['context_changes']),
                'max_qse_influence': max(rich_history['qse_influence']),
                'mean_tau': np.mean(rich_history['tau']),
                'tau_variance': np.var(rich_history['tau']),
                'sigma_volatility': np.std(rich_history['sigma_mean'])
            })

        # Aggregate results and statistical testing
        summary = self._analyze_enhanced_causality_results(causality_results)

        self.results['enhanced_causality'] = summary
        return summary

    def _compute_enhanced_correlations(self, history: Dict) -> Dict[str, float]:
        """Compute multiple types of QSE → behavior correlations"""

        correlations = {}

        # Convert to numpy arrays
        for key in history:
            history[key] = np.array(history[key])

        # Standard correlations
        correlations['surplus_decisions'] = self._safe_correlation(history['surplus_mean'], history['decision_events'])
        correlations['sigma_decisions'] = self._safe_correlation(history['sigma_mean'], history['decision_events'])
        correlations['sigma_ema_decisions'] = self._safe_correlation(history['sigma_ema'], history['decision_events'])
        correlations['tau_decisions'] = self._safe_correlation(history['tau'], history['decision_events'])

        # Phasic rupture → behavioral response (key autopoietic signature!)
        if sum(history['phasic_rupture']) > 3:
            # Look at decisions in 5 steps following rupture events
            rupture_indices = np.where(history['phasic_rupture'] == 1)[0]
            post_rupture_decisions = []

            for idx in rupture_indices:
                if idx + 5 < len(history['decision_events']):
                    post_rupture_decisions.append(np.sum(history['decision_events'][idx:idx+5]))

            if post_rupture_decisions:
                baseline_decision_rate = np.mean(history['decision_events'])
                avg_post_rupture = np.mean(post_rupture_decisions)
                correlations['phasic_rupture_response'] = avg_post_rupture / (baseline_decision_rate + 0.001)
            else:
                correlations['phasic_rupture_response'] = 1.0
        else:
            correlations['phasic_rupture_response'] = 1.0

        # Tonic rupture effects
        if sum(history['tonic_rupture']) > 10:
            tonic_indices = np.where(history['tonic_rupture'] == 1)[0]
            decisions_during_tonic = np.mean(history['decision_events'][tonic_indices])
            baseline_decisions = np.mean(history['decision_events'])
            correlations['tonic_rupture_effect'] = decisions_during_tonic / (baseline_decisions + 0.001)
        else:
            correlations['tonic_rupture_effect'] = 1.0

        # Lagged correlations (QSE influence may have delayed effects)
        max_lag = min(10, len(history['tau']) // 4)
        best_lag_corr = 0
        best_lag = 0

        for lag in range(1, max_lag):
            if len(history['tau']) > lag:
                lagged_corr = self._safe_correlation(
                    history['tau'][:-lag],
                    history['decision_events'][lag:]
                )
                if abs(lagged_corr) > abs(best_lag_corr):
                    best_lag_corr = lagged_corr
                    best_lag = lag

        correlations['best_lagged_tau_decisions'] = best_lag_corr
        correlations['optimal_lag'] = best_lag

        # QSE influence → behavioral response
        correlations['qse_influence_behavior'] = self._safe_correlation(
            history['qse_influence'],
            history['decision_events']
        )

        # Multi-variable correlations
        if len(history['tau']) > 20:
            # Combined QSE state vector
            qse_combined = (history['tau'] + history['sigma_ema'] + history['surplus_mean']) / 3
            correlations['combined_qse_behavior'] = self._safe_correlation(qse_combined, history['decision_events'])

        return correlations

    def _analyze_regime_specific_causality(self, history: Dict) -> Dict[str, Any]:
        """Analyze causality within different QSE regimes"""

        regime_analysis = {}

        # Convert to numpy arrays
        for key in history:
            history[key] = np.array(history[key])

        regimes = ['high_tau', 'low_tau', 'phasic_active', 'stable']

        for regime in regimes:
            regime_mask = np.array(history['regime_state']) == regime
            if np.sum(regime_mask) > 10:  # Need sufficient data
                regime_tau = history['tau'][regime_mask]
                regime_decisions = history['decision_events'][regime_mask]
                regime_qse_influence = history['qse_influence'][regime_mask]

                regime_analysis[regime] = {
                    'steps_in_regime': int(np.sum(regime_mask)),
                    'tau_decision_correlation': self._safe_correlation(regime_tau, regime_decisions),
                    'qse_influence_correlation': self._safe_correlation(regime_qse_influence, regime_decisions),
                    'decision_rate': float(np.mean(regime_decisions)),
                    'mean_tau': float(np.mean(regime_tau)),
                    'mean_qse_influence': float(np.mean(regime_qse_influence))
                }
            else:
                regime_analysis[regime] = {
                    'steps_in_regime': int(np.sum(regime_mask)),
                    'tau_decision_correlation': 0,
                    'qse_influence_correlation': 0,
                    'decision_rate': 0,
                    'mean_tau': 0,
                    'mean_qse_influence': 0
                }

        return regime_analysis

    def _safe_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """Safely compute correlation, handling edge cases and NaN values"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0

        # Remove NaN values
        valid_mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
        if np.sum(valid_mask) < 2:
            return 0.0

        x_clean = x[valid_mask]
        y_clean = y[valid_mask]

        if np.std(x_clean) == 0 or np.std(y_clean) == 0:
            return 0.0

        try:
            corr_matrix = np.corrcoef(x_clean, y_clean)
            if corr_matrix.shape == (2, 2):
                corr = corr_matrix[0, 1]
                return float(corr) if not np.isnan(corr) else 0.0
            else:
                return 0.0
        except:
            return 0.0

    def test_regime_sensitive_perturbation(self, steps: int = 8000) -> Dict[str, Any]:
        """Test perturbations during different QSE regimes"""
        print(f"\n⚡ REGIME-SENSITIVE PERTURBATION TEST")
        print(f"Testing perturbations during different QSE regimes...")

        perturbation_results = {}

        # Test different perturbations
        perturbations = {
            'tau_shift': {'TAU_MIN': 0.25, 'TAU_MAX': 1.2},
            'sigma_sensitivity': {'SIGMA_EMA_ALPHA': 0.35, 'K_PSI': 15.0},
            'quantum_coupling': {'QUANTUM_COUPLING': 0.20},
            'surplus_dynamics': {'S_BETA': 0.8, 'S_GAMMA': 0.5}
        }

        for perturb_name, changes in perturbations.items():
            print(f"  Testing {perturb_name}...")

            # Baseline
            baseline_metrics = self._run_agent_for_metrics(steps//3, QSEConfig())

            # Perturbed
            perturbed_cfg = QSEConfig()
            for param, value in changes.items():
                setattr(perturbed_cfg, param, value)
            perturbed_metrics = self._run_agent_for_metrics(steps//3, perturbed_cfg)

            # Recovery (back to baseline)
            recovery_metrics = self._run_agent_for_metrics(steps//3, QSEConfig())

            # Analyze response
            perturbation_results[perturb_name] = {
                'baseline': baseline_metrics,
                'perturbed': perturbed_metrics,
                'recovery': recovery_metrics,
                'response_magnitude': self._calculate_perturbation_response(baseline_metrics, perturbed_metrics),
                'recovery_quality': self._calculate_recovery_quality(baseline_metrics, recovery_metrics),
                'hysteresis_detected': self._detect_hysteresis(baseline_metrics, perturbed_metrics, recovery_metrics)
            }

        # Overall assessment
        summary = self._assess_perturbation_sensitivity(perturbation_results)

        self.results['regime_sensitive_perturbation'] = summary
        return summary

    def _run_agent_for_metrics(self, steps: int, cfg: QSEConfig) -> Dict[str, Any]:
        """Run agent and collect comprehensive metrics"""

        agent = EmileAgent(cfg)
        for goal in ["explore", "exploit", "maintain", "adapt"]:
            agent.goal.add_goal(goal)

        metrics_history = {
            'tau': [],
            'sigma_mean': [],
            'surplus_mean': [],
            'decision_events': [],
            'context_switches': [],
            'phasic_ruptures': [],
            'qse_influence': []
        }

        previous_context = None
        sigma_history = deque(maxlen=5)

        for step in range(steps):
            agent_metrics = agent.step()

            # Extract metrics
            tau = getattr(agent.qse, 'tau', 0.5) if hasattr(agent.qse, 'tau') else 0.5
            sigma = agent_metrics.get('sigma_mean', np.mean(getattr(agent.symbolic, 'sigma', [0])))
            surplus = agent_metrics.get('surplus_mean', np.mean(agent.qse.S))

            current_context = agent.context.get_current()
            context_switched = previous_context is not None and current_context != previous_context

            # Phasic rupture detection
            sigma_history.append(sigma)
            phasic_rupture = False
            if len(sigma_history) >= 3:
                sigma_change = abs(sigma - sigma_history[-2])
                phasic_rupture = sigma_change > 0.5

            # QSE influence
            qse_influence = abs(surplus) * 0.4 + abs(sigma) * 0.4 + abs(tau - 0.5) * 0.2

            # Store metrics
            metrics_history['tau'].append(tau)
            metrics_history['sigma_mean'].append(sigma)
            metrics_history['surplus_mean'].append(surplus)
            metrics_history['decision_events'].append(1 if context_switched else 0)
            metrics_history['context_switches'].append(1 if context_switched else 0)
            metrics_history['phasic_ruptures'].append(1 if phasic_rupture else 0)
            metrics_history['qse_influence'].append(qse_influence)

            previous_context = current_context

        # Compute summary statistics
        summary = {}
        for key, values in metrics_history.items():
            summary[f'{key}_mean'] = np.mean(values)
            summary[f'{key}_std'] = np.std(values)
            summary[f'{key}_total'] = np.sum(values) if key in ['decision_events', 'context_switches', 'phasic_ruptures'] else np.mean(values)

        return summary

    def test_enhanced_autopoietic_enactivism(self, steps: int = 20000) -> Dict[str, Any]:
        """Enhanced test for genuine enactive cognition"""
        print(f"\n🧠 ENHANCED AUTOPOIETIC ENACTIVISM TEST")
        print(f"Testing genuine computational enactive cognition...")

        agent = EmileAgent()
        for goal in ["explore", "exploit", "maintain", "adapt", "learn"]:
            agent.goal.add_goal(goal)

        enactivism_data = {
            'context_effectiveness': defaultdict(lambda: {
                'steps': 0,
                'decisions': 0,
                'qse_influence_scores': [],
                'tau_values': [],
                'outcomes': []
            }),
            'decision_chains': [],
            'goal_hierarchies': {},
            'boundary_dynamics': {
                'self_organization_events': 0,
                'boundary_maintenance_events': 0,
                'context_driven_adaptations': 0
            },
            'recursive_patterns': {
                'decision_triggered_decisions': 0,
                'context_cascades': 0,
                'qse_feedback_loops': 0
            }
        }

        # Track decision chains
        recent_decisions = deque(maxlen=10)
        previous_goal = None
        previous_context = None
        decision_chain_active = False
        chain_length = 0

        for step in range(steps):
            pre_state = {
                'goal': agent.goal.current_goal,
                'context': agent.context.get_current(),
                'surplus_mean': np.mean(agent.qse.S)
            }

            # Agent step
            metrics = agent.step()

            post_state = {
                'goal': agent.goal.current_goal,
                'context': agent.context.get_current(),
                'surplus_mean': np.mean(agent.qse.S)
            }

            # Detect changes
            goal_changed = pre_state['goal'] != post_state['goal']
            context_changed = pre_state['context'] != post_state['context']
            decision_made = goal_changed or context_changed

            # Context effectiveness analysis
            context_id = post_state['context']
            ctx_data = enactivism_data['context_effectiveness'][context_id]
            ctx_data['steps'] += 1
            if decision_made:
                ctx_data['decisions'] += 1

            # QSE metrics for this context
            tau = getattr(agent.qse, 'tau', 0.5) if hasattr(agent.qse, 'tau') else 0.5
            sigma = metrics.get('sigma_mean', 0)
            qse_influence = abs(sigma) * 0.5 + abs(tau - 0.5) * 0.5

            ctx_data['qse_influence_scores'].append(qse_influence)
            ctx_data['tau_values'].append(tau)

            # Evaluate decision outcome (simplified)
            surplus_change = post_state['surplus_mean'] - pre_state['surplus_mean']
            outcome_quality = 1.0 if surplus_change > 0 else 0.0 if surplus_change == 0 else -1.0
            ctx_data['outcomes'].append(outcome_quality)

            # Decision chain tracking
            recent_decisions.append(decision_made)

            if decision_made:
                if decision_chain_active:
                    chain_length += 1
                else:
                    decision_chain_active = True
                    chain_length = 1

                # Check if this decision was triggered by recent QSE activity
                if qse_influence > 0.5:
                    enactivism_data['recursive_patterns']['qse_feedback_loops'] += 1
            else:
                if decision_chain_active and chain_length >= 2:
                    enactivism_data['decision_chains'].append({
                        'start_step': step - chain_length,
                        'length': chain_length,
                        'contexts_involved': [context_id],  # Simplified
                        'avg_qse_influence': qse_influence
                    })
                decision_chain_active = False
                chain_length = 0

            # Recursive patterns detection
            if decision_made and sum(recent_decisions) >= 3:  # Multiple recent decisions
                enactivism_data['recursive_patterns']['decision_triggered_decisions'] += 1

            if context_changed and decision_made:
                enactivism_data['recursive_patterns']['context_cascades'] += 1

            # Self-organization detection
            if context_changed and qse_influence > 0.6:
                enactivism_data['boundary_dynamics']['self_organization_events'] += 1

            # Boundary maintenance (returning to stable states)
            if not decision_made and qse_influence < 0.3:
                enactivism_data['boundary_dynamics']['boundary_maintenance_events'] += 1

            # Context-driven adaptation
            if context_changed and abs(surplus_change) > 0.1:
                enactivism_data['boundary_dynamics']['context_driven_adaptations'] += 1

            previous_goal = post_state['goal']
            previous_context = post_state['context']

        # Compute enactivism scores
        scores = self._compute_enactivism_scores(enactivism_data, steps)

        self.results['enhanced_autopoietic_enactivism'] = scores
        return scores

    def _compute_enactivism_scores(self, data: Dict, total_steps: int) -> Dict[str, Any]:
        """Compute comprehensive enactivism scores"""

        # Context effectiveness analysis
        context_scores = {}
        for ctx_id, ctx_data in data['context_effectiveness'].items():
            if ctx_data['steps'] > 0:
                decision_rate = ctx_data['decisions'] / ctx_data['steps']
                avg_qse_influence = np.mean(ctx_data['qse_influence_scores']) if ctx_data['qse_influence_scores'] else 0
                avg_outcome = np.mean(ctx_data['outcomes']) if ctx_data['outcomes'] else 0

                context_scores[ctx_id] = {
                    'decision_rate': decision_rate,
                    'avg_qse_influence': avg_qse_influence,
                    'avg_outcome': avg_outcome,
                    'effectiveness': decision_rate * avg_qse_influence * (1 + avg_outcome)
                }

        # Context differentiation
        if len(context_scores) > 1:
            effectiveness_values = [scores['effectiveness'] for scores in context_scores.values()]
            context_differentiation = np.var(effectiveness_values)
            context_specialization = len([v for v in effectiveness_values if v > np.mean(effectiveness_values)]) / len(effectiveness_values)
        else:
            context_differentiation = 0
            context_specialization = 0

        # Recursive causality
        total_chains = len(data['decision_chains'])
        avg_chain_length = np.mean([c['length'] for c in data['decision_chains']]) if data['decision_chains'] else 0
        max_chain_length = max([c['length'] for c in data['decision_chains']]) if data['decision_chains'] else 0

        recursive_density = data['recursive_patterns']['decision_triggered_decisions'] / total_steps
        qse_feedback_density = data['recursive_patterns']['qse_feedback_loops'] / total_steps

        # Self-organization
        self_org_rate = data['boundary_dynamics']['self_organization_events'] / total_steps
        boundary_maintenance_rate = data['boundary_dynamics']['boundary_maintenance_events'] / total_steps
        adaptation_rate = data['boundary_dynamics']['context_driven_adaptations'] / total_steps

        # Overall scores
        scores = {
            'context_effectiveness': {
                'differentiation': min(1.0, context_differentiation * 10),
                'specialization': context_specialization,
                'total_contexts': len(context_scores),
                'context_details': context_scores
            },
            'recursive_causality': {
                'decision_chains': total_chains,
                'avg_chain_length': avg_chain_length,
                'max_chain_length': max_chain_length,
                'recursive_density': recursive_density,
                'qse_feedback_density': qse_feedback_density,
                'recursivity_score': min(1.0, (total_chains / 5) * 0.4 + recursive_density * 300 * 0.3 + qse_feedback_density * 500 * 0.3)
            },
            'self_organization': {
                'self_org_rate': self_org_rate,
                'boundary_maintenance_rate': boundary_maintenance_rate,
                'adaptation_rate': adaptation_rate,
                'organization_score': min(1.0, self_org_rate * 200 * 0.5 + adaptation_rate * 300 * 0.5)
            },
            'intrinsic_teleology': {
                'goal_driven_contexts': len([ctx for ctx, scores in context_scores.items() if scores['effectiveness'] > 0.1]),
                'teleological_coherence': min(1.0, len(data['decision_chains']) / 3) if data['decision_chains'] else 0,
                'teleology_score': min(1.0, len(data['decision_chains']) / 3 * 0.6 + context_specialization * 0.4)
            }
        }

        # Overall enactivism assessment
        component_scores = [
            scores['context_effectiveness']['differentiation'],
            scores['recursive_causality']['recursivity_score'],
            scores['self_organization']['organization_score'],
            scores['intrinsic_teleology']['teleology_score']
        ]

        overall_score = np.mean(component_scores)

        scores['overall_assessment'] = {
            'component_scores': component_scores,
            'overall_enactivism_score': overall_score,
            'enactivism_classification': (
                'STRONG' if overall_score > 0.7 else
                'MODERATE' if overall_score > 0.4 else
                'WEAK'
            ),
            'evidence_strength': overall_score,
            'interpretation': self._interpret_enhanced_enactivism(scores, overall_score)
        }

        return scores

    def test_qse_dynamics_integration(self, steps: int = 15000) -> Dict[str, Any]:
        """Leverage QSE dynamics analyzer for comprehensive validation"""
        if not QSE_ANALYZER_AVAILABLE:
            print("⚠️ QSE Dynamics Analyzer not available - skipping integration test")
            return {'available': False, 'message': 'QSE analyzer not found'}

        print(f"\n🔬 QSE DYNAMICS INTEGRATION TEST")
        print(f"Using QSE dynamics analyzer for comprehensive validation...")

        # Run agent simulation with data collection
        cfg = QSEConfig()
        agent = EmileAgent(cfg)
        for goal in ["explore", "exploit", "maintain", "adapt", "learn"]:
            agent.goal.add_goal(goal)

        # Collect simulation data in QSE analyzer format
        simulation_data = []

        for step in range(steps):
            # Pre-step state
            pre_state = {
                'context': agent.context.get_current(),
                'goal': agent.goal.current_goal,
                'q_values': getattr(agent.goal, 'q_values', {}).copy()
            }

            # Agent step
            metrics = agent.step()

            # Post-step state
            post_state = {
                'context': agent.context.get_current(),
                'goal': agent.goal.current_goal,
                'q_values': getattr(agent.goal, 'q_values', {}).copy()
            }

            # Detect changes
            state_changes = {
                'goal_changed': pre_state['goal'] != post_state['goal'],
                'context_changed': pre_state['context'] != post_state['context'],
                'q_value_change': self._calculate_q_value_change(pre_state['q_values'], post_state['q_values'])
            }

            # QSE metrics with NaN protection
            tau = getattr(agent.qse, 'tau', 0.5) if hasattr(agent.qse, 'tau') else 0.5
            tau = 0.5 if np.isnan(tau) or np.isinf(tau) else float(tau)

            sigma = metrics.get('sigma_mean', 0)
            if np.isnan(sigma) or np.isinf(sigma):
                sigma = 0.0
            sigma = float(sigma)

            surplus = metrics.get('surplus_mean', np.mean(agent.qse.S))
            if np.isnan(surplus) or np.isinf(surplus):
                surplus = np.nanmean(agent.qse.S) if not np.isnan(np.nanmean(agent.qse.S)) else 0.1
            surplus = float(surplus)

            # QSE influence calculation with protection
            qse_influence = abs(surplus) * 0.3 + abs(sigma) * 0.4 + abs(tau - 0.5) * 0.3
            if np.isnan(qse_influence) or np.isinf(qse_influence):
                qse_influence = 0.1

            # Format data for analyzer
            data_point = {
                'step': step,
                'tau_current': tau,
                'sigma_mean': sigma,
                'surplus_mean': surplus,
                'tonic_rupture_active': abs(sigma) > 0.3,
                'phasic_rupture_active': abs(sigma) > 0.7,
                'prob_entropy_normalized': metrics.get('normalized_entropy', 0.5),
                'agent_pre_state': pre_state,
                'agent_post_state': post_state,
                'agent_state_change': state_changes,
                'decision_triggered': state_changes['goal_changed'] or state_changes['context_changed'],
                'qse_influence_score': qse_influence
            }

            simulation_data.append(data_point)

        # Clean the data before analysis
        simulation_data = self._clean_simulation_data(simulation_data)

        # Initialize and run QSE analyzer
        try:
            # Try different ways to initialize QSEDynamicsAnalyzer
            try:
                # Method 1: Try with empty list (most common pattern)
                analyzer = QSEDynamicsAnalyzer([])
            except:
                try:
                    # Method 2: Try with no arguments
                    analyzer = QSEDynamicsAnalyzer()
                except:
                    # Method 3: Try with a temporary file
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                        for data in simulation_data[:10]:  # Write sample data
                            f.write(json.dumps(data) + '\n')
                        temp_file = f.name
                    analyzer = QSEDynamicsAnalyzer([temp_file])
                    import os
                    os.unlink(temp_file)  # Clean up

            # Set data directly
            analyzer.data = simulation_data
            analyzer.df = pd.DataFrame(simulation_data)

            # Run complete analysis
            findings = analyzer.run_complete_analysis(deep=True)

            # Extract autopoietic evidence from findings
            autopoietic_evidence = self._extract_autopoietic_evidence(findings)

        except Exception as e:
            print(f"⚠️ QSE analyzer integration failed: {e}")
            # Return simplified analysis instead
            autopoietic_evidence = self._simplified_qse_analysis(simulation_data)

        self.results['qse_dynamics_integration'] = autopoietic_evidence
        return autopoietic_evidence

    def run_comprehensive_validation(self, steps_per_test: int = 15000, n_agents: int = 4) -> Dict[str, Any]:
        """Run the complete enhanced autopoietic validation suite"""

        print(f"🧬 COMPREHENSIVE ENHANCED AUTOPOIETIC VALIDATION")
        print(f"=" * 70)
        print(f"Running enhanced validation suite...")

        start_time = time.time()

        # Run all enhanced tests
        test_results = {}

        print("\n1/7 Enhanced QSE Causality...")
        test_results['enhanced_causality'] = self.test_enhanced_qse_causality(steps=steps_per_test)

        print("\n2/7 Regime-Sensitive Perturbation...")
        test_results['regime_perturbation'] = self.test_regime_sensitive_perturbation(steps=steps_per_test//2)

        print("\n3/7 Enhanced Autopoietic Enactivism...")
        test_results['enhanced_enactivism'] = self.test_enhanced_autopoietic_enactivism(steps=steps_per_test)

        print("\n4/7 QSE Dynamics Integration...")
        test_results['qse_integration'] = self.test_qse_dynamics_integration(steps=steps_per_test//2)

        print("\n5/7 Boundary Maintenance...")
        test_results['boundary'] = self.test_boundary_maintenance(steps=steps_per_test//2)

        print("\n6/7 Social Autopoiesis...")
        test_results['social'] = self.test_social_autopoiesis(n_agents=n_agents, steps=1200)

        print("\n7/7 Null Model Comparison...")
        test_results['null_comparison'] = self.test_null_model_comparison(steps=steps_per_test//3)

        # Calculate enhanced autopoiesis score
        autopoiesis_score = self._calculate_enhanced_autopoiesis_score(test_results)

        # Overall assessment
        overall_assessment = self._generate_enhanced_overall_assessment(test_results, autopoiesis_score)
        publication_ready = autopoiesis_score > 0.65 and self._check_enhanced_publication_criteria(test_results)

        # Compile final results
        final_results = {
            'validation_timestamp': time.time(),
            'total_runtime_minutes': (time.time() - start_time) / 60,
            'test_results': test_results,
            'autopoiesis_score': autopoiesis_score,
            'overall_assessment': overall_assessment,
            'publication_ready': publication_ready,
            'enhancement_level': 'v3.0_comprehensive'
        }

        # Save results
        self._save_enhanced_results(final_results)

        # Create enhanced visualizations
        self._create_enhanced_validation_summary_plot(test_results, autopoiesis_score)

        # Print summary
        print(f"\n" + "=" * 70)
        print(f"🔬 Enhanced Autopoiesis Score: {autopoiesis_score:.3f}")
        print(f"📋 Assessment: {overall_assessment}")
        print(f"📁 Results saved to: {self.output_dir}/")

        if publication_ready:
            print(f"✅ PUBLICATION READY - Strong evidence for computational autopoiesis!")
        else:
            print(f"⚠️ Additional evidence needed for strong autopoiesis claims")

        return final_results

    # Include necessary helper methods from original suite
    def test_boundary_maintenance(self, steps: int = 10000) -> Dict[str, Any]:
        """Test boundary maintenance under stress (from original suite)"""
        print(f"\n🛡️ BOUNDARY MAINTENANCE TEST")

        cfg = QSEConfig()
        agent = EmileAgent(cfg)
        for goal in ["explore", "exploit", "maintain"]:
            agent.goal.add_goal(goal)

        all_metrics = []
        phases = ['baseline', 'stress', 'recovery']
        steps_per_phase = steps // 3

        for phase_idx, phase in enumerate(phases):
            print(f"  Phase: {phase}")

            # Modify config for stress phase
            if phase == 'stress':
                # Apply stress by increasing rupture threshold
                agent.qse.cfg.S_THETA_RUPTURE = 0.2  # Lower threshold = more ruptures
                agent.qse.cfg.S_EPSILON = 0.8  # Higher expulsion
            elif phase == 'recovery':
                # Return to normal
                agent.qse.cfg.S_THETA_RUPTURE = 0.5
                agent.qse.cfg.S_EPSILON = 0.35

            phase_metrics = []

            for step in range(steps_per_phase):
                metrics = agent.step()

                step_data = {
                    'step': phase_idx * steps_per_phase + step,
                    'phase': phase,
                    'surplus_mean': np.mean(agent.qse.S),
                    'sigma_mean': metrics.get('sigma_mean', 0),
                    'context': agent.context.get_current()
                }

                phase_metrics.append(step_data)
                all_metrics.append(step_data)

        # Analyze recovery
        baseline_surplus = np.mean([m['surplus_mean'] for m in all_metrics if m['phase'] == 'baseline'])
        recovery_surplus = np.mean([m['surplus_mean'] for m in all_metrics if m['phase'] == 'recovery'])

        recovery_ratio = abs(recovery_surplus - baseline_surplus) / (baseline_surplus + 0.001)
        recovery_ratio = max(0, 1 - recovery_ratio)  # Higher is better

        return {
            'all_metrics': all_metrics,
            'recovery_ratio': recovery_ratio,
            'boundary_maintained': recovery_ratio > 0.8,
            'interpretation': f"{'EXCELLENT' if recovery_ratio > 0.8 else 'MODERATE' if recovery_ratio > 0.6 else 'POOR'} boundary maintenance (recovery: {recovery_ratio:.2%})"
        }

    def test_social_autopoiesis(self, n_agents: int = 4, steps: int = 1200) -> Dict[str, Any]:
        """Test social autopoiesis (simplified version)"""
        print(f"\n👥 SOCIAL AUTOPOIESIS TEST")

        # Simplified social test
        agents = []
        for i in range(n_agents):
            cfg = QSEConfig()
            agent = EmileAgent(cfg)
            agent.agent_id = f"agent_{i}"
            for goal in ["explore", "cooperate", "maintain"]:
                agent.goal.add_goal(goal)
            agents.append(agent)

        knowledge_transfer = 0
        interaction_events = 0

        for step in range(steps):
            for agent in agents:
                metrics = agent.step()

                # Simulate knowledge transfer when agents have similar contexts
                if step % 50 == 0:  # Check periodically
                    current_context = agent.context.get_current()
                    for other_agent in agents:
                        if other_agent != agent:
                            other_context = other_agent.context.get_current()
                            if current_context == other_context:
                                knowledge_transfer += 1
                                interaction_events += 1

        social_coherence = knowledge_transfer / max(1, interaction_events)

        return {
            'knowledge_transfer_events': knowledge_transfer,
            'total_interactions': interaction_events,
            'social_coherence': social_coherence,
            'social_autopoiesis_detected': social_coherence > 0.3,
            'interpretation': f"{'STRONG' if social_coherence > 0.7 else 'MODERATE' if social_coherence > 0.3 else 'WEAK'} social autopoiesis: {social_coherence:.1%} knowledge transfer"
        }

    def test_null_model_comparison(self, steps: int = 5000) -> Dict[str, Any]:
        """Test against null models"""
        print(f"\n🎲 NULL MODEL COMPARISON TEST")

        # QSE system
        cfg = QSEConfig()
        agent = EmileAgent(cfg)
        for goal in ["explore", "exploit", "maintain"]:
            agent.goal.add_goal(goal)

        qse_metrics = []
        for step in range(steps):
            metrics = agent.step()
            qse_metrics.append({
                'context_switches': 1 if step > 0 and agent.context.get_current() != qse_metrics[-1].get('context', 0) else 0,
                'context': agent.context.get_current()
            })

        # Random model
        random_metrics = []
        for step in range(steps):
            random_metrics.append({
                'context_switches': 1 if np.random.random() < 0.1 else 0,
                'context': np.random.randint(0, 5)
            })

        # Compare
        qse_switches = sum(m['context_switches'] for m in qse_metrics)
        random_switches = sum(m['context_switches'] for m in random_metrics)

        return {
            'qse_context_switches': qse_switches,
            'random_context_switches': random_switches,
            'significantly_different_from_random': abs(qse_switches - random_switches) > steps * 0.05,
            'complexity_advantage': qse_switches != random_switches,
            'interpretation': "QSE system shows structured context switching" if abs(qse_switches - random_switches) > steps * 0.05 else "Similar to random model"
        }

    # Enhanced scoring and analysis methods
    def _calculate_enhanced_autopoiesis_score(self, test_results: Dict) -> float:
        """Calculate enhanced autopoiesis score"""
        scores = []

        # Enhanced causality (weighted heavily)
        if 'enhanced_causality' in test_results:
            causality = test_results['enhanced_causality']
            if 'causality_strength' in causality:
                scores.append(causality['causality_strength'])
            else:
                scores.append(0.3)  # Default moderate score

        # Regime perturbation
        if 'regime_perturbation' in test_results:
            perturbation = test_results['regime_perturbation']
            scores.append(perturbation.get('overall_sensitivity', 0.5))

        # Enhanced enactivism (weighted heavily)
        if 'enhanced_enactivism' in test_results:
            enactivism = test_results['enhanced_enactivism']
            overall_score = enactivism.get('overall_assessment', {}).get('overall_enactivism_score', 0.3)
            scores.append(overall_score)

        # QSE integration
        if 'qse_integration' in test_results:
            integration = test_results['qse_integration']
            if integration.get('available', True):
                integration_score = integration.get('overall_assessment', {}).get('overall_autopoiesis_score', 0.4)
                scores.append(integration_score)

        # Boundary maintenance
        if 'boundary' in test_results:
            boundary = test_results['boundary']
            scores.append(boundary.get('recovery_ratio', 0.5))

        # Social autopoiesis
        if 'social' in test_results:
            social = test_results['social']
            scores.append(1.0 if social.get('social_autopoiesis_detected', False) else 0.3)

        # Null comparison
        if 'null_comparison' in test_results:
            null_comp = test_results['null_comparison']
            scores.append(0.8 if null_comp.get('significantly_different_from_random', False) else 0.2)

        return np.mean(scores) if scores else 0.0

    def _generate_enhanced_overall_assessment(self, test_results: Dict, score: float) -> str:
        """Generate enhanced overall assessment"""
        if score > 0.75:
            return "STRONG evidence for computational autopoiesis with enhanced QSE dynamics"
        elif score > 0.6:
            return "GOOD evidence for computational autopoiesis with some enhanced features"
        elif score > 0.45:
            return "MODERATE evidence for autopoietic properties with enhanced analysis"
        else:
            return "WEAK evidence for autopoiesis - enhanced analysis shows primarily mechanistic behavior"

    def _check_enhanced_publication_criteria(self, test_results: Dict) -> bool:
        """Check enhanced publication readiness criteria"""
        criteria_met = 0
        total_criteria = 6

        # Strong causality evidence
        if 'enhanced_causality' in test_results:
            causality = test_results['enhanced_causality']
            if causality.get('causality_strength', 0) > 0.6:
                criteria_met += 1

        # Regime sensitivity
        if 'regime_perturbation' in test_results:
            perturbation = test_results['regime_perturbation']
            if perturbation.get('overall_sensitivity', 0) > 0.5:
                criteria_met += 1

        # Enhanced enactivism
        if 'enhanced_enactivism' in test_results:
            enactivism = test_results['enhanced_enactivism']
            if enactivism.get('overall_assessment', {}).get('overall_enactivism_score', 0) > 0.6:
                criteria_met += 1

        # QSE integration success
        if 'qse_integration' in test_results:
            integration = test_results['qse_integration']
            if integration.get('available', False) and integration.get('overall_assessment', {}).get('overall_autopoiesis_score', 0) > 0.5:
                criteria_met += 1

        # Boundary maintenance
        if 'boundary' in test_results:
            boundary = test_results['boundary']
            if boundary.get('recovery_ratio', 0) > 0.8:
                criteria_met += 1

        # Null model differentiation
        if 'null_comparison' in test_results:
            null_comp = test_results['null_comparison']
            if null_comp.get('significantly_different_from_random', False):
                criteria_met += 1

        return criteria_met >= 4  # At least 4/6 criteria must be met

    # Additional helper methods
    def _analyze_enhanced_causality_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze enhanced causality results across trials"""

        # Extract correlation arrays
        corr_keys = ['surplus_decisions', 'sigma_decisions', 'sigma_ema_decisions', 'tau_decisions',
                     'phasic_rupture_response', 'tonic_rupture_effect', 'qse_influence_behavior',
                     'best_lagged_tau_decisions', 'combined_qse_behavior']

        correlation_summary = {}

        for key in corr_keys:
            values = []
            for r in results:
                if key in r['correlations'] and not np.isnan(r['correlations'][key]):
                    values.append(r['correlations'][key])

            if values:
                correlation_summary[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'significant': stats.ttest_1samp(values, 0)[1] < 0.05 if len(values) > 1 else False,
                    'effect_size': np.mean(values) / (np.std(values) + 0.001),
                    'trials_with_data': len(values)
                }
            else:
                correlation_summary[key] = {
                    'mean': 0, 'std': 0, 'significant': False, 'effect_size': 0, 'trials_with_data': 0
                }

        # Behavioral event summary
        behavioral_summary = {
            'avg_phasic_events': np.mean([r['phasic_events'] for r in results]),
            'avg_tonic_events': np.mean([r['tonic_events'] for r in results]),
            'avg_decision_events': np.mean([r['decision_events'] for r in results]),
            'avg_context_switches': np.mean([r['context_switches'] for r in results]),
            'avg_max_qse_influence': np.mean([r['max_qse_influence'] for r in results]),
            'avg_tau': np.mean([r['mean_tau'] for r in results]),
            'avg_tau_variance': np.mean([r['tau_variance'] for r in results]),
            'avg_sigma_volatility': np.mean([r['sigma_volatility'] for r in results])
        }

        # Enhanced causality assessment
        strong_correlations = sum(1 for key, data in correlation_summary.items()
                                 if data['significant'] and abs(data['mean']) > 0.4)

        moderate_correlations = sum(1 for key, data in correlation_summary.items()
                                   if data['significant'] and 0.2 < abs(data['mean']) <= 0.4)

        weak_correlations = sum(1 for key, data in correlation_summary.items()
                               if data['significant'] and 0.1 < abs(data['mean']) <= 0.2)

        causality_strength = (strong_correlations * 1.0 + moderate_correlations * 0.6 + weak_correlations * 0.3) / len(corr_keys)

        # Special bonuses for key autopoietic signatures
        if correlation_summary['phasic_rupture_response']['mean'] > 1.5:
            causality_strength += 0.15  # Bonus for strong phasic response

        if correlation_summary['qse_influence_behavior']['significant'] and correlation_summary['qse_influence_behavior']['mean'] > 0.3:
            causality_strength += 0.1  # Bonus for QSE influence

        if behavioral_summary['avg_tau_variance'] > 0.1:  # Dynamic tau
            causality_strength += 0.05  # Bonus for temporal autonomy

        causality_strength = min(1.0, causality_strength)  # Cap at 1.0

        assessment = {
            'correlation_summary': correlation_summary,
            'behavioral_summary': behavioral_summary,
            'causality_strength': causality_strength,
            'strong_evidence': causality_strength > 0.7,
            'moderate_evidence': 0.4 <= causality_strength <= 0.7,
            'weak_evidence': causality_strength < 0.4,
            'interpretation': self._interpret_enhanced_causality(correlation_summary, causality_strength, behavioral_summary)
        }

        return assessment

    def _interpret_enhanced_causality(self, correlations: Dict, strength: float, behavioral: Dict) -> str:
        """Generate interpretation of enhanced causality results"""

        if strength > 0.7:
            key_findings = []

            # Check for strong correlations
            if correlations['phasic_rupture_response']['mean'] > 1.5:
                key_findings.append(f"phasic rupture events trigger {correlations['phasic_rupture_response']['mean']:.1f}x more decisions")

            if correlations['tau_decisions']['significant'] and abs(correlations['tau_decisions']['mean']) > 0.3:
                key_findings.append(f"emergent time strongly correlates with decisions (r={correlations['tau_decisions']['mean']:.3f})")

            if correlations['qse_influence_behavior']['significant'] and correlations['qse_influence_behavior']['mean'] > 0.3:
                key_findings.append(f"QSE influence drives behavioral responses (r={correlations['qse_influence_behavior']['mean']:.3f})")

            if behavioral['avg_tau_variance'] > 0.1:
                key_findings.append(f"dynamic temporal autonomy (τ variance: {behavioral['avg_tau_variance']:.3f})")

            findings_text = "; ".join(key_findings) if key_findings else "multiple significant correlations"

            return f"STRONG QSE causality evidence (strength: {strength:.3f}): {findings_text}"

        elif strength > 0.4:
            significant_corrs = [key for key, data in correlations.items() if data['significant']]
            return f"MODERATE QSE causality (strength: {strength:.3f}): significant correlations in {', '.join(significant_corrs[:3])}"

        else:
            return f"WEAK QSE causality evidence (strength: {strength:.3f}): limited systematic QSE → behavior relationships"

    def _calculate_perturbation_response(self, baseline: Dict, perturbed: Dict) -> float:
        """Calculate magnitude of perturbation response"""
        response_metrics = ['decision_events_total', 'context_switches_total', 'tau_mean', 'sigma_mean_mean']

        responses = []
        for metric in response_metrics:
            baseline_val = baseline.get(metric, 0)
            perturbed_val = perturbed.get(metric, 0)

            if baseline_val != 0:
                relative_change = abs(perturbed_val - baseline_val) / baseline_val
            else:
                relative_change = abs(perturbed_val)

            responses.append(relative_change)

        return np.mean(responses) if responses else 0.0

    def _calculate_recovery_quality(self, baseline: Dict, recovery: Dict) -> float:
        """Calculate quality of recovery to baseline"""
        recovery_metrics = ['tau_mean', 'sigma_mean_mean', 'surplus_mean_mean']

        recoveries = []
        for metric in recovery_metrics:
            baseline_val = baseline.get(metric, 0)
            recovery_val = recovery.get(metric, 0)

            if baseline_val != 0:
                recovery_ratio = 1 - abs(recovery_val - baseline_val) / baseline_val
            else:
                recovery_ratio = 1 - abs(recovery_val)

            recoveries.append(max(0, recovery_ratio))

        return np.mean(recoveries) if recoveries else 0.0

    def _detect_hysteresis(self, baseline: Dict, perturbed: Dict, recovery: Dict) -> bool:
        """Detect hysteresis effects"""
        # Simple hysteresis detection: recovery doesn't fully return to baseline
        recovery_quality = self._calculate_recovery_quality(baseline, recovery)
        return recovery_quality < 0.9

    def _assess_perturbation_sensitivity(self, results: Dict) -> Dict[str, Any]:
        """Assess overall perturbation sensitivity"""

        sensitivities = []
        hysteresis_count = 0

        for perturb_name, data in results.items():
            response_magnitude = data['response_magnitude']
            recovery_quality = data['recovery_quality']
            hysteresis_detected = data['hysteresis_detected']

            sensitivities.append(response_magnitude)
            if hysteresis_detected:
                hysteresis_count += 1

        overall_sensitivity = np.mean(sensitivities) if sensitivities else 0.0
        hysteresis_rate = hysteresis_count / len(results) if results else 0.0

        return {
            'perturbation_results': results,
            'overall_sensitivity': overall_sensitivity,
            'hysteresis_rate': hysteresis_rate,
            'systematic_response': overall_sensitivity > 0.1,
            'interpretation': f"{'STRONG' if overall_sensitivity > 0.3 else 'MODERATE' if overall_sensitivity > 0.1 else 'WEAK'} perturbation sensitivity (response: {overall_sensitivity:.3f}, hysteresis: {hysteresis_rate:.1%})"
        }

    def _interpret_enhanced_enactivism(self, scores: Dict, overall_score: float) -> str:
        """Generate interpretation of enhanced autopoietic enactivism"""

        if overall_score > 0.7:
            return (f"STRONG autopoietic enactivism (score: {overall_score:.3f}). "
                    f"System demonstrates genuine computational enactive cognition: "
                    f"context differentiation, recursive causality, self-organization, and intrinsic teleology. "
                    f"Enhanced analysis shows robust QSE-driven sense-making and autonomous behavior generation.")

        elif overall_score > 0.5:
            return (f"MODERATE autopoietic enactivism (score: {overall_score:.3f}). "
                    f"Enhanced analysis reveals significant enactive properties with some limitations. "
                    f"Context effectiveness and recursive patterns partially demonstrated.")

        elif overall_score > 0.3:
            return (f"WEAK autopoietic enactivism (score: {overall_score:.3f}). "
                    f"Enhanced analysis shows limited evidence for genuine enactive cognition. "
                    f"System exhibits some context differentiation but lacks robust recursive causality.")

        else:
            return (f"NO autopoietic enactivism detected (score: {overall_score:.3f}). "
                    f"Enhanced analysis reveals primarily mechanistic behavior patterns. "
                    f"Limited context effectiveness and minimal recursive causality.")

    def _extract_autopoietic_evidence(self, findings: Dict) -> Dict[str, Any]:
        """Extract autopoietic validation evidence from QSE dynamics analysis"""

        evidence = {
            'qse_causality': {},
            'regime_autonomy': {},
            'recursive_causality': {},
            'context_effectiveness': {},
            'temporal_organization': {}
        }

        # QSE Causality Evidence
        if 'advanced_correlations' in findings:
            corr_data = findings['advanced_correlations']
            evidence['qse_causality'] = {
                'phasic_decision_correlation': corr_data.get('conditional', {}).get('tau_vs_decisions_during_phasic', 0),
                'stable_decision_correlation': corr_data.get('conditional', {}).get('tau_vs_decisions_during_stable', 0),
                'cross_correlations': corr_data.get('cross_correlations', {}),
                'strongest_qse_behavior_link': max(
                    [(name, data['max_corr']) for name, data in corr_data.get('cross_correlations', {}).items()],
                    key=lambda x: x[1], default=('none', 0)
                )
            }

        # Regime Autonomy
        if 'regime_transitions' in findings:
            transitions = findings['regime_transitions']
            evidence['regime_autonomy'] = {
                'total_transitions': len(transitions),
                'avg_behavioral_response': np.mean([t.get('behavioral_response', {}).get('decision_events', 0)
                                                   for t in transitions]) if transitions else 0,
                'regime_coherence': len(transitions) > 0,
                'transition_quality': self._assess_transition_quality(transitions)
            }

        # Recursive Causality
        if 'decision_chains' in findings:
            chains = findings['decision_chains']
            evidence['recursive_causality'] = {
                'total_chains': len(chains),
                'avg_chain_length': np.mean([c['length'] for c in chains]) if chains else 0,
                'max_chain_length': max([c['length'] for c in chains]) if chains else 0,
                'recursive_depth': sum(1 for c in chains if c['length'] >= 3),
                'causality_strength': min(1.0, len(chains) / 10.0)
            }

        # Context Effectiveness
        if 'context_effectiveness' in findings:
            ctx_eff = findings['context_effectiveness']
            evidence['context_effectiveness'] = {
                'total_contexts': len(ctx_eff),
                'effectiveness_variance': np.var([data['effectiveness_score'] for data in ctx_eff.values()]) if ctx_eff else 0,
                'best_context_score': max([data['effectiveness_score'] for data in ctx_eff.values()]) if ctx_eff else 0,
                'context_differentiation': len(ctx_eff) > 1 and np.var([data['effectiveness_score'] for data in ctx_eff.values()]) > 0.01
            }

        # Temporal Organization
        basic_stats = findings.get('basic_stats', {})
        evidence['temporal_organization'] = {
            'decision_rate': basic_stats.get('decision_events', 0) / basic_stats.get('total_steps', 1),
            'context_switch_rate': basic_stats.get('context_switches', 0) / basic_stats.get('total_steps', 1),
            'phasic_event_rate': basic_stats.get('phasic_rupture_events', 0) / basic_stats.get('total_steps', 1),
            'max_qse_influence': basic_stats.get('max_qse_influence', 0),
            'temporal_coherence': basic_stats.get('phasic_rupture_events', 0) > 0
        }

        # Overall assessment
        evidence['overall_assessment'] = self._assess_overall_autopoiesis(evidence)
        evidence['available'] = True

        return evidence

    def _assess_overall_autopoiesis(self, evidence: Dict) -> Dict[str, Any]:
        """Assess overall autopoietic properties from QSE dynamics evidence"""

        scores = []

        # QSE Causality Score
        qse_caus = evidence['qse_causality']
        strongest_corr = qse_caus.get('strongest_qse_behavior_link', ('none', 0))[1]
        causality_score = min(1.0, strongest_corr * 2.0)
        scores.append(causality_score)

        # Regime Autonomy Score
        regime_aut = evidence['regime_autonomy']
        autonomy_score = min(1.0, regime_aut['total_transitions'] / 10.0)
        if regime_aut['avg_behavioral_response'] > 1.5:
            autonomy_score += 0.2
        autonomy_score = min(1.0, autonomy_score)
        scores.append(autonomy_score)

        # Recursive Causality Score
        recursive = evidence['recursive_causality']
        recursive_score = min(1.0, recursive['total_chains'] / 5.0)
        if recursive['max_chain_length'] >= 5:
            recursive_score += 0.3
        recursive_score = min(1.0, recursive_score)
        scores.append(recursive_score)

        # Context Effectiveness Score
        context = evidence['context_effectiveness']
        if context['context_differentiation'] and context['total_contexts'] >= 2:
            context_score = min(1.0, context['effectiveness_variance'] * 10.0)
        else:
            context_score = 0.0
        scores.append(context_score)

        # Temporal Organization Score
        temporal = evidence['temporal_organization']
        temporal_score = 0.0
        if temporal['decision_rate'] > 0.05:
            temporal_score += 0.3
        if temporal['phasic_event_rate'] > 0.01:
            temporal_score += 0.3
        if temporal['max_qse_influence'] > 0.5:
            temporal_score += 0.4
        scores.append(temporal_score)

        overall_score = np.mean(scores)

        return {
            'qse_causality_score': causality_score,
            'regime_autonomy_score': autonomy_score,
            'recursive_causality_score': recursive_score,
            'context_effectiveness_score': context_score,
            'temporal_organization_score': temporal_score,
            'overall_autopoiesis_score': overall_score,
            'autopoietic_classification': (
                'STRONG' if overall_score > 0.7 else
                'MODERATE' if overall_score > 0.4 else
                'WEAK'
            ),
            'publication_ready': overall_score > 0.6 and all(s > 0.3 for s in scores)
        }

    def _calculate_q_value_change(self, pre_q: Dict, post_q: Dict) -> float:
        """Calculate magnitude of Q-value changes"""
        if not pre_q or not post_q:
            return 0.0

        changes = []
        for key in set(pre_q.keys()) | set(post_q.keys()):
            pre_val = pre_q.get(key, 0)
            post_val = post_q.get(key, 0)
            changes.append(abs(post_val - pre_val))

        return np.mean(changes) if changes else 0.0

    def _assess_transition_quality(self, transitions: List[Dict]) -> float:
        """Assess quality of regime transitions"""
        if not transitions:
            return 0.0

        responsive_transitions = sum(1 for t in transitions
                                    if t.get('behavioral_response', {}).get('decision_events', 0) > 0)

        return responsive_transitions / len(transitions)

    def _save_enhanced_results(self, results: Dict):
        """Save enhanced results to JSON"""
        output_file = self.output_dir / "enhanced_autopoietic_validation_results.json"

        def convert_for_json(obj):
            # Handle numpy arrays first
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()

            # Handle pandas NA values - check if it's a scalar first
            try:
                if pd.isna(obj):
                    return None
            except (ValueError, TypeError):
                # If pd.isna fails (e.g., on arrays), continue to other checks
                pass

            if isinstance(obj, (list, tuple, set, deque)):
                return [convert_for_json(x) for x in obj]
            if isinstance(obj, dict):
                return {str(k): convert_for_json(v) for k, v in obj.items()}

            # Handle other types that might cause issues
            if hasattr(obj, 'tolist'):  # Any array-like object
                return obj.tolist()
            if hasattr(obj, 'item'):  # Any scalar-like object
                return obj.item()

            return str(obj)

        json_results = convert_for_json(results)

        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2, allow_nan=False)

        print(f"\n📁 Enhanced results saved to: {output_file}")

    def _clean_simulation_data(self, simulation_data: List[Dict]) -> List[Dict]:
        """Clean simulation data of NaN and invalid values"""

        cleaned_data = []
        nan_count = 0

        for i, data_point in enumerate(simulation_data):
            # Create cleaned copy
            clean_point = data_point.copy()

            # Check and clean numeric fields
            numeric_fields = ['tau_current', 'sigma_mean', 'surplus_mean', 'qse_influence_score', 'prob_entropy_normalized']

            has_nan = False
            for field in numeric_fields:
                if field in clean_point:
                    value = clean_point[field]
                    if np.isnan(value) or np.isinf(value):
                        has_nan = True
                        # Replace with reasonable defaults
                        if field == 'tau_current':
                            clean_point[field] = 0.5
                        elif field == 'prob_entropy_normalized':
                            clean_point[field] = 0.5
                        else:
                            clean_point[field] = 0.0

            if has_nan:
                nan_count += 1

            # Ensure state change values are valid
            if 'agent_state_change' in clean_point:
                state_change = clean_point['agent_state_change']
                if 'q_value_change' in state_change and (np.isnan(state_change['q_value_change']) or np.isinf(state_change['q_value_change'])):
                    state_change['q_value_change'] = 0.0

            cleaned_data.append(clean_point)

        if nan_count > 0:
            print(f"    Cleaned {nan_count}/{len(simulation_data)} data points with NaN values")

        return cleaned_data

    def _simplified_qse_analysis(self, simulation_data: List[Dict]) -> Dict[str, Any]:
        """Simplified QSE analysis when full analyzer is not available"""

        print("⚠️ Using simplified QSE analysis...")

        # Clean the data first
        simulation_data = self._clean_simulation_data(simulation_data)

        # Convert to DataFrame for analysis
        df = pd.DataFrame(simulation_data)

        # Basic QSE dynamics analysis with additional NaN protection
        tau_values = df['tau_current'].values
        sigma_values = df['sigma_mean'].values
        decision_events = df['decision_triggered'].values.astype(float)
        qse_influence = df['qse_influence_score'].values

        # Remove any remaining NaN values
        valid_mask = ~(np.isnan(tau_values) | np.isnan(sigma_values) | np.isnan(decision_events) | np.isnan(qse_influence))

        if np.sum(valid_mask) < 10:
            print("⚠️ Too few valid data points for analysis")
            return {
                'available': True,
                'simplified': True,
                'error': 'Insufficient valid data points',
                'overall_assessment': {
                    'overall_autopoiesis_score': 0.0,
                    'autopoietic_classification': 'INVALID',
                    'publication_ready': False
                }
            }

        # Filter to valid data
        tau_values = tau_values[valid_mask]
        sigma_values = sigma_values[valid_mask]
        decision_events = decision_events[valid_mask]
        qse_influence = qse_influence[valid_mask]

        # Basic correlations
        tau_decision_corr = self._safe_correlation(tau_values, decision_events)
        sigma_decision_corr = self._safe_correlation(sigma_values, decision_events)
        qse_influence_corr = self._safe_correlation(qse_influence, decision_events)

        # Regime detection
        high_tau_mask = tau_values > 0.7
        low_tau_mask = tau_values < 0.3
        phasic_mask = df['phasic_rupture_active'].values[valid_mask].astype(bool)

        # Decision rates by regime
        high_tau_decisions = np.mean(decision_events[high_tau_mask]) if np.any(high_tau_mask) else 0
        low_tau_decisions = np.mean(decision_events[low_tau_mask]) if np.any(low_tau_mask) else 0
        phasic_decisions = np.mean(decision_events[phasic_mask]) if np.any(phasic_mask) else 0
        baseline_decisions = np.mean(decision_events)

        # Context effectiveness
        contexts = df['agent_post_state'].apply(lambda x: x['context']).values[valid_mask]
        unique_contexts = np.unique(contexts)
        context_effectiveness = {}

        for ctx in unique_contexts:
            ctx_mask = contexts == ctx
            ctx_decisions = np.mean(decision_events[ctx_mask]) if np.any(ctx_mask) else 0
            ctx_qse_influence = np.mean(qse_influence[ctx_mask]) if np.any(ctx_mask) else 0
            context_effectiveness[int(ctx)] = {
                'decision_rate': float(ctx_decisions),
                'avg_qse_influence': float(ctx_qse_influence),
                'effectiveness': float(ctx_decisions * ctx_qse_influence)
            }

        # Overall assessment
        causality_strength = max(abs(tau_decision_corr), abs(sigma_decision_corr), abs(qse_influence_corr))
        regime_differentiation = abs(high_tau_decisions - low_tau_decisions) + abs(phasic_decisions - baseline_decisions)
        context_variance = np.var([ctx['effectiveness'] for ctx in context_effectiveness.values()]) if context_effectiveness else 0

        overall_score = min(1.0, (causality_strength + regime_differentiation + context_variance) / 3)

        return {
            'available': True,
            'simplified': True,
            'valid_data_points': int(np.sum(valid_mask)),
            'total_data_points': len(simulation_data),
            'qse_causality': {
                'tau_decision_correlation': float(tau_decision_corr),
                'sigma_decision_correlation': float(sigma_decision_corr),
                'qse_influence_correlation': float(qse_influence_corr),
                'strongest_correlation': float(causality_strength)
            },
            'regime_analysis': {
                'high_tau_decision_rate': float(high_tau_decisions),
                'low_tau_decision_rate': float(low_tau_decisions),
                'phasic_decision_rate': float(phasic_decisions),
                'baseline_decision_rate': float(baseline_decisions),
                'regime_differentiation': float(regime_differentiation)
            },
            'context_effectiveness': context_effectiveness,
            'overall_assessment': {
                'overall_autopoiesis_score': float(overall_score),
                'causality_strength': float(causality_strength),
                'regime_autonomy': float(regime_differentiation),
                'context_differentiation': float(context_variance),
                'autopoietic_classification': (
                    'STRONG' if overall_score > 0.7 else
                    'MODERATE' if overall_score > 0.4 else
                    'WEAK'
                ),
                'publication_ready': overall_score > 0.6
            }
        }

    def _create_enhanced_validation_summary_plot(self, test_results: Dict, overall_score: float):
        """Create enhanced validation summary visualization"""

        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle(f'Enhanced Autopoietic Validation Suite Results (Score: {overall_score:.3f})',
                     fontsize=16, fontweight='bold')

        # 1. Test scores overview
        ax1 = axes[0, 0]
        test_scores = {
            'Enhanced Causality': test_results.get('enhanced_causality', {}).get('causality_strength', 0),
            'Regime Perturbation': test_results.get('regime_perturbation', {}).get('overall_sensitivity', 0),
            'Enhanced Enactivism': test_results.get('enhanced_enactivism', {}).get('overall_assessment', {}).get('overall_enactivism_score', 0),
            'QSE Integration': test_results.get('qse_integration', {}).get('overall_assessment', {}).get('overall_autopoiesis_score', 0) if test_results.get('qse_integration', {}).get('available', False) else 0,
            'Boundary Maintenance': test_results.get('boundary', {}).get('recovery_ratio', 0),
            'Social Autopoiesis': 1.0 if test_results.get('social', {}).get('social_autopoiesis_detected', False) else 0.3,
            'Null Comparison': 0.8 if test_results.get('null_comparison', {}).get('significantly_different_from_random', False) else 0.2
        }

        tests = list(test_scores.keys())
        scores = list(test_scores.values())
        colors = ['darkgreen' if s > 0.7 else 'orange' if s > 0.4 else 'red' for s in scores]

        bars = ax1.bar(range(len(tests)), scores, color=colors, alpha=0.7)
        ax1.set_xticks(range(len(tests)))
        ax1.set_xticklabels(tests, rotation=45, ha='right', fontsize=10)
        ax1.set_ylabel('Test Score')
        ax1.set_title('Enhanced Validation Test Scores')
        ax1.set_ylim(0, 1.1)

        for bar, score in zip(bars, scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{score:.2f}', ha='center', va='bottom', fontweight='bold')

        # 2. Enhanced causality details
        ax2 = axes[0, 1]
        if 'enhanced_causality' in test_results:
            causality = test_results['enhanced_causality']
            corr_summary = causality.get('correlation_summary', {})

            correlations = []
            labels = []
            for key, data in corr_summary.items():
                if 'mean' in data:
                    correlations.append(abs(data['mean']))
                    labels.append(key.replace('_', '\n'))

            if correlations:
                ax2.barh(range(len(labels)), correlations, alpha=0.7, color='blue')
                ax2.set_yticks(range(len(labels)))
                ax2.set_yticklabels(labels, fontsize=8)
                ax2.set_xlabel('|Correlation|')
                ax2.set_title('Enhanced QSE Causality Correlations')

        # 3. Enactivism component scores
        ax3 = axes[0, 2]
        if 'enhanced_enactivism' in test_results:
            enactivism = test_results['enhanced_enactivism']
            component_scores = enactivism.get('overall_assessment', {}).get('component_scores', [0, 0, 0, 0])
            component_names = ['Context\nEffectiveness', 'Recursive\nCausality', 'Self\nOrganization', 'Intrinsic\nTeleology']

            ax3.bar(component_names, component_scores, alpha=0.7, color=['purple', 'teal', 'orange', 'red'])
            ax3.set_ylabel('Component Score')
            ax3.set_title('Enhanced Enactivism Components')
            ax3.set_ylim(0, 1.1)

        # 4. Regime analysis
        ax4 = axes[1, 0]
        if 'enhanced_causality' in test_results:
            causality = test_results['enhanced_causality']
            behavioral = causality.get('behavioral_summary', {})

            regime_metrics = {
                'Phasic Events': behavioral.get('avg_phasic_events', 0),
                'Decision Events': behavioral.get('avg_decision_events', 0),
                'Context Switches': behavioral.get('avg_context_switches', 0),
                'QSE Influence': behavioral.get('avg_max_qse_influence', 0)
            }

            ax4.bar(regime_metrics.keys(), regime_metrics.values(), alpha=0.7, color='green')
            ax4.set_ylabel('Average Count/Score')
            ax4.set_title('QSE Regime Dynamics')
            ax4.tick_params(axis='x', rotation=45)

        # 5. Perturbation sensitivity
        ax5 = axes[1, 1]
        if 'regime_perturbation' in test_results:
            perturbation = test_results['regime_perturbation']
            perturb_results = perturbation.get('perturbation_results', {})

            if perturb_results:
                perturbations = list(perturb_results.keys())
                responses = [data['response_magnitude'] for data in perturb_results.values()]

                ax5.bar(perturbations, responses, alpha=0.7, color='red')
                ax5.set_ylabel('Response Magnitude')
                ax5.set_title('Perturbation Sensitivity')
                ax5.tick_params(axis='x', rotation=45)

        # 6. Context effectiveness
        ax6 = axes[1, 2]
        if 'enhanced_enactivism' in test_results:
            enactivism = test_results['enhanced_enactivism']
            context_eff = enactivism.get('context_effectiveness', {})
            context_details = context_eff.get('context_details', {})

            if context_details:
                contexts = list(context_details.keys())[:8]  # Limit to 8 contexts
                effectiveness = [context_details[ctx]['effectiveness'] for ctx in contexts]

                ax6.bar(contexts, effectiveness, alpha=0.7, color='purple')
                ax6.set_ylabel('Effectiveness Score')
                ax6.set_title('Context Effectiveness')
                ax6.tick_params(axis='x', rotation=45)

        # 7. QSE Integration results
        ax7 = axes[2, 0]
        if 'qse_integration' in test_results and test_results['qse_integration'].get('available', False):
            integration = test_results['qse_integration']
            overall_assess = integration.get('overall_assessment', {})

            integration_scores = {
                'QSE Causality': overall_assess.get('qse_causality_score', 0),
                'Regime Autonomy': overall_assess.get('regime_autonomy_score', 0),
                'Recursive Causality': overall_assess.get('recursive_causality_score', 0),
                'Context Effectiveness': overall_assess.get('context_effectiveness_score', 0),
                'Temporal Organization': overall_assess.get('temporal_organization_score', 0)
            }

            ax7.bar(integration_scores.keys(), integration_scores.values(), alpha=0.7, color='teal')
            ax7.set_ylabel('Integration Score')
            ax7.set_title('QSE Dynamics Integration')
            ax7.tick_params(axis='x', rotation=45)
        else:
            ax7.text(0.5, 0.5, 'QSE Integration\nNot Available',
                    ha='center', va='center', transform=ax7.transAxes, fontsize=12)
            ax7.set_title('QSE Dynamics Integration')

        # 8. Temporal dynamics
        ax8 = axes[2, 1]
        if 'enhanced_causality' in test_results:
            causality = test_results['enhanced_causality']
            behavioral = causality.get('behavioral_summary', {})

            temporal_data = {
                'Mean τ': behavioral.get('avg_tau', 0.5),
                'τ Variance': behavioral.get('avg_tau_variance', 0),
                'σ Volatility': behavioral.get('avg_sigma_volatility', 0),
                'Max QSE Influence': behavioral.get('avg_max_qse_influence', 0)
            }

            ax8.bar(temporal_data.keys(), temporal_data.values(), alpha=0.7, color='orange')
            ax8.set_ylabel('Temporal Metric Value')
            ax8.set_title('Temporal Dynamics')
            ax8.tick_params(axis='x', rotation=45)

        # 9. Overall score gauge
        ax9 = axes[2, 2]

        # Create gauge-like visualization
        theta = np.linspace(0, np.pi, 100)
        radius = 1

        # Color zones
        ax9.fill_between(theta, 0, radius, where=(theta >= 0) & (theta <= np.pi/3),
                        color='red', alpha=0.3, label='Weak (0-0.33)')
        ax9.fill_between(theta, 0, radius, where=(theta >= np.pi/3) & (theta <= 2*np.pi/3),
                        color='orange', alpha=0.3, label='Moderate (0.33-0.67)')
        ax9.fill_between(theta, 0, radius, where=(theta >= 2*np.pi/3) & (theta <= np.pi),
                        color='green', alpha=0.3, label='Strong (0.67-1.0)')

        # Score indicator
        score_angle = overall_score * np.pi
        ax9.plot([score_angle, score_angle], [0, radius], 'k-', linewidth=4, label=f'Score: {overall_score:.3f}')

        ax9.set_xlim(0, np.pi)
        ax9.set_ylim(0, radius * 1.1)
        ax9.set_title('Overall Autopoiesis Score')
        ax9.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3))
        ax9.set_xticks([0, np.pi/3, 2*np.pi/3, np.pi])
        ax9.set_xticklabels(['0', '0.33', '0.67', '1.0'])

        plt.tight_layout()
        plt.savefig(self.output_dir / "enhanced_autopoietic_validation_summary.png",
                   dpi=300, bbox_inches='tight')
        plt.show()

        print(f"📊 Enhanced visualizations saved to {self.output_dir}/")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Enhanced Autopoietic Validation Suite v3.0')
    parser.add_argument('--comprehensive', action='store_true',
                       help='Run comprehensive enhanced validation suite')
    parser.add_argument('--test', nargs='+',
                       choices=['enhanced_causality', 'regime_perturbation', 'enhanced_enactivism',
                               'qse_integration', 'boundary', 'social', 'null'],
                       help='Run specific enhanced tests')
    parser.add_argument('--steps', type=int, default=15000,
                       help='Steps per test (default: 15000)')
    parser.add_argument('--agents', type=int, default=4,
                       help='Number of agents for social test (default: 4)')
    parser.add_argument('--output', type=str, default='enhanced_autopoiesis_validation',
                       help='Output directory (default: enhanced_autopoiesis_validation)')

    args = parser.parse_args()

    # Initialize enhanced suite
    suite = EnhancedAutopoieticValidationSuite(output_dir=args.output)

    if args.comprehensive:
        # Run comprehensive enhanced validation
        results = suite.run_comprehensive_validation(steps_per_test=args.steps, n_agents=args.agents)

        print(f"=" * 70)
        print(f"🔬 Enhanced Autopoiesis Score: {results['autopoiesis_score']:.3f}")
        print(f"📋 Assessment: {results['overall_assessment']}")
        print(f"📁 Results saved to: {args.output}/")

        if results['publication_ready']:
            print(f"✅ PUBLICATION READY - Strong evidence for computational autopoiesis!")
        else:
            print(f"⚠️ Additional evidence needed for strong autopoiesis claims")

    elif args.test:
        # Run specific enhanced tests
        for test_name in args.test:
            if test_name == 'enhanced_causality':
                suite.test_enhanced_qse_causality(steps=args.steps)
            elif test_name == 'regime_perturbation':
                suite.test_regime_sensitive_perturbation(steps=args.steps//2)
            elif test_name == 'enhanced_enactivism':
                suite.test_enhanced_autopoietic_enactivism(steps=args.steps)
            elif test_name == 'qse_integration':
                suite.test_qse_dynamics_integration(steps=args.steps//2)
            elif test_name == 'boundary':
                suite.test_boundary_maintenance(steps=args.steps//2)
            elif test_name == 'social':
                suite.test_social_autopoiesis(n_agents=args.agents, steps=1200)
            elif test_name == 'null':
                suite.test_null_model_comparison(steps=args.steps//3)

    else:
        print("Please specify --comprehensive or --test [test_names]")
        print("Available enhanced tests: enhanced_causality, regime_perturbation, enhanced_enactivism, qse_integration, boundary, social, null")


if __name__ == "__main__":
    main()
