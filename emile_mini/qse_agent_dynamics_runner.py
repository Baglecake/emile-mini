#!/usr/bin/env python3
"""
QSE Agent Dynamics Runner - Autopoiesis ↔ Behavior Analysis
==========================================================

Simultaneous capture of:
- Deep QSE autopoietic dynamics (tonic/phasic rupture, tau emergence, sigma curvature)
- Agent cognitive/behavioral responses (decisions, actions, context switches)
- Cross-correlations between autopoiesis and behavior
- Causal relationships: How dynamics shape decisions

Usage:
    python qse_agent_dynamics_runner.py --steps 10000 --agent embodied --environment maze
    python qse_agent_dynamics_runner.py --steps 50000 --agent cognitive --analysis deep
"""

import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Import QSE metrics and agent components
from emile_mini.qse_core_metric_runner_c import QSEMetricsCollector
from emile_mini.config import QSEConfig
from emile_mini.agent import EmileAgent
from emile_mini.embodied_qse_emile import EmbodiedQSEAgent, EmbodiedEnvironment
from emile_mini.qse_core import calculate_symbolic_fields, calculate_emergent_time


class QSEAgentDynamicsRunner:
    """Simultaneous QSE dynamics and agent behavior analysis"""

    def __init__(self, agent_type: str = "cognitive", environment_type: str = "basic"):
        self.agent_type = agent_type
        self.environment_type = environment_type
        self._step_count: int = 0

        # QSE metrics collector
        self.cfg = QSEConfig()
        self.qse_collector = QSEMetricsCollector(self.cfg)

        # Agent and environment
        self.agent = self._create_agent()
        self.environment = self._create_environment()

        # Data storage
        self.dynamics_history = []  # QSE + Agent combined data
        self.decision_events = []   # Significant decision points
        self.correlation_matrices = {}

        # Analysis results
        self.dynamics_patterns = {}
        self.behavior_patterns = {}
        self.causal_relationships = {}

    def _create_agent(self):
        """Create agent based on type specification"""

        if self.agent_type == "cognitive":
            agent = EmileAgent(self.cfg)
            # Add diverse goals for rich behavior
            goals = ["explore", "exploit", "maintain", "adapt", "learn", "create"]
            for goal in goals:
                agent.goal.add_goal(goal)

        elif self.agent_type == "embodied":
            agent = EmbodiedQSEAgent(self.cfg)
            # FIX: Add goals explicitly
            goals = ["explore", "navigate", "survive", "forage", "rest", "investigate"]
            for goal in goals:
                agent.goal.add_goal(goal)
            # Embodied goals already added in constructor

        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")

        print(f"🤖 Created {self.agent_type} agent")
        return agent

    def _create_environment(self):
        """Create environment based on type specification"""

        if self.environment_type == "basic":
            return None  # Cognitive agent doesn't need environment

        elif self.environment_type == "embodied":
            env = EmbodiedEnvironment(size=15)
            if hasattr(self.agent, 'body'):
                # Place agent in environment
                self.agent.body.state.position = (7, 7)

            print(f"🌍 Created embodied environment (15x15)")
            return env

        else:
            raise ValueError(f"Unknown environment type: {self.environment_type}")

    def run_dynamics_analysis(self, steps: int = 10000, dt: float = 0.01,
                            output_file: str = None, seed: int = 42) -> Dict[str, Any]:
        """Run simultaneous QSE dynamics and agent behavior analysis"""

        print(f"🔬 QSE-AGENT DYNAMICS ANALYSIS")
        print(f"=" * 50)
        print(f"Agent: {self.agent_type}")
        print(f"Environment: {self.environment_type}")
        print(f"Steps: {steps}, dt: {dt}, Seed: {seed}")

        # Set seeds for reproducibility
        np.random.seed(seed)

        # Prepare output
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        # Run simultaneous dynamics
        print(f"\n⚡ Running simultaneous QSE dynamics + agent behavior...")
        self._run_simultaneous_dynamics(steps, dt, output_file)

        # Analyze patterns
        print(f"\n🧠 Analyzing dynamics-behavior patterns...")
        self._analyze_dynamics_patterns()

        # Analyze correlations
        print(f"\n🔗 Computing dynamics-behavior correlations...")
        self._analyze_dynamics_correlations()

        # Extract causal relationships
        print(f"\n🎯 Identifying causal relationships...")
        self._identify_causal_relationships()

        # Generate comprehensive report
        print(f"\n📋 Generating comprehensive report...")
        results = self._generate_dynamics_report()

        # Create visualizations
        print(f"\n📊 Creating dynamics visualizations...")
        self._create_dynamics_visualizations()

        return results

    def _run_simultaneous_dynamics(self, steps: int, dt: float, output_file: str):
        """Run QSE dynamics and agent behavior simultaneously"""

        sigma_prev = None
        start_time = time.time()

        # Open output file if specified
        file_handle = None
        if output_file:
            file_handle = open(output_file, 'w')

        try:
            for step in range(steps):
                # === QSE DYNAMICS ===
                psi, phi, sigma = calculate_symbolic_fields(self.qse_collector.engine.S, self.cfg)

                # Calculate emergent time
                if sigma_prev is not None:
                    tau = calculate_emergent_time(sigma, sigma_prev, self.cfg)
                else:
                    tau = self.cfg.TAU_MAX

                # Collect QSE metrics
                qse_metrics = self.qse_collector.collect_step_metrics(sigma, tau)

                # === AGENT BEHAVIOR ===
                pre_agent_state = self._capture_agent_state()

                if self.agent_type == "cognitive":
                    external_input = self._generate_external_input(step, qse_metrics)
                    agent_result = self.agent.step(dt=dt, external_input=external_input)

                elif self.agent_type == "embodied":
                    # Built-in target scheduler for richer behavior (every 200 steps)
                    if hasattr(self.agent, "receive_memory_cue") and (self._step_count % 200 == 0):
                        quadrants = ["NE", "NW", "SE", "SW", "C"]
                        new_quad = quadrants[(self._step_count // 200) % len(quadrants)]
                        self.agent.receive_memory_cue({
                            "type": "navigation_cue",
                            "target_quadrant": new_quad,
                            "instruction": f"Navigate to {new_quad} quadrant",
                            "priority": "high",
                        })

                    # Embodied step with environment
                    agent_result = self.agent.embodied_step(self.environment, dt=dt)

                else:
                    # Fallback (shouldn’t happen if args validated)
                    agent_result = {"action": "idle"}

                post_agent_state = self._capture_agent_state()

                # === COMBINE DATA ===
                combined_step_data = {
                    "step": step,
                    "timestamp": time.time(),
                    "dt": dt,

                    # QSE DYNAMICS
                    **qse_metrics,

                    # AGENT BEHAVIOR
                    "agent_pre_state": pre_agent_state,
                    "agent_post_state": post_agent_state,
                    "agent_result": self._sanitize_agent_result(agent_result),
                    "agent_state_change": self._calculate_agent_state_change(pre_agent_state, post_agent_state),

                    # DYNAMICS-BEHAVIOR LINKS
                    "decision_triggered": self._detect_decision_event(pre_agent_state, post_agent_state, qse_metrics),
                    "qse_influence_score": self._calculate_qse_influence(qse_metrics, pre_agent_state, post_agent_state),
                }

                # --- Analyzer-friendly flat aliases ---
                post = combined_step_data.get("agent_post_state", {}) or {}
                pre  = combined_step_data.get("agent_pre_state", {}) or {}
                chg  = combined_step_data.get("agent_state_change", {}) or {}
                res  = combined_step_data.get("agent_result", {}) or {}

                # Safe context handling (robust to None/strings)
                ctx_val = post.get("context", pre.get("context", 0))
                try:
                    ctx_int = int(ctx_val)
                except (TypeError, ValueError):
                    ctx_int = 0

                combined_step_data.update({
                    "behavior": str(res.get("action", res.get("behavior", "idle"))),
                    "goal": str(post.get("goal", pre.get("goal", "none"))),
                    "context": ctx_int,
                    "context_changed": 1 if chg.get("context_changed") else 0,
                    "qse_influence": float(combined_step_data.get("qse_influence_score", 0.0)),
                    "q_value_change": float(chg.get("q_value_change", 0.0)),
                    "decision_events": 1.0 if combined_step_data.get("decision_triggered") else 0.0,
                })

                # Ensure analyzer sees a 'tau' key
                if "tau" not in combined_step_data and "tau_current" in combined_step_data:
                    combined_step_data["tau"] = combined_step_data["tau_current"]

                # Normalize strings (never None)
                if combined_step_data["goal"] in (None, "None"):
                    combined_step_data["goal"] = "none"
                if combined_step_data["behavior"] in (None, "None"):
                    combined_step_data["behavior"] = "idle"

                # Store data
                self.dynamics_history.append(combined_step_data)

                # Write to file
                if file_handle:
                    file_handle.write(json.dumps(combined_step_data, default=str) + "\n")
                    if step % 200 == 0:
                        file_handle.flush()

                # Progress reporting
                if step % 1000 == 0:
                    tau_print = combined_step_data.get("tau", combined_step_data.get("tau_current", float("nan")))
                    print(
                        f"  Step {step:5d}: τ={tau_print:.3f}, "
                        f"goal={combined_step_data['goal']}, "
                        f"context={combined_step_data['context']}, "
                        f"influence={combined_step_data['qse_influence']:.3f}"
                    )

                # bump internal counter once per loop
                self._step_count += 1

                # keep previous sigma for next tau computation
                sigma_prev = sigma.copy()

        finally:
            if file_handle:
                file_handle.close()

        elapsed = time.time() - start_time
        print(f"  ✅ Collected {len(self.dynamics_history)} simultaneous QSE-agent steps in {elapsed:.2f}s")


    def _capture_agent_state(self) -> Dict[str, Any]:
        """Capture current agent state"""

        state = {
            'goal': self._get_current_goal(),
            'context': self._get_current_context(),
            'q_values': dict(self.agent.goal.q_values) if hasattr(self.agent.goal, 'q_values') else {},
            'memory_working_size': len(self.agent.memory.get_working()) if hasattr(self.agent, 'memory') else 0,
            'memory_episodic_size': len(self.agent.memory.get_episodic()) if hasattr(self.agent, 'memory') else 0
        }

        # Add embodied state if applicable
        if hasattr(self.agent, 'body'):
            state.update({
                'position': getattr(self.agent.body.state, 'position', None),
                'energy': self.agent.body.state.energy,
                'orientation': getattr(self.agent.body.state, 'orientation', 0.0)
            })

        return state

    def _get_current_goal(self) -> str:
        """Get current agent goal"""
        if hasattr(self.agent, 'goal') and hasattr(self.agent.goal, 'current_goal'):
            return str(self.agent.goal.current_goal)
        return "unknown"

    def _get_current_context(self) -> int:
        """Get current agent context"""
        if hasattr(self.agent, 'context'):
            return int(self.agent.context.get_current())
        return 0

    def _generate_external_input(self, step: int, qse_metrics: Dict) -> Dict[str, Any]:
        """Generate external input for cognitive agent based on QSE state"""

        # Provide rewards based on QSE health
        reward = 0.0

        # Reward healthy autopoietic dynamics
        if qse_metrics.get('tonic_rupture_active', False):
            reward += 0.1  # Background aliveness bonus

        if qse_metrics.get('phasic_rupture_active', False):
            reward += 0.2  # Regime transition bonus

        # Entropy-based rewards
        entropy = qse_metrics.get('prob_entropy_normalized', 0.5)
        if 0.7 <= entropy <= 0.95:
            reward += 0.15  # Healthy entropy range

        # Periodic environmental changes
        if step % 100 == 0:
            reward += 0.5  # Periodic challenges

        return {'reward': reward}

    def _sanitize_agent_result(self, result: Any) -> Dict[str, Any]:
        """Sanitize agent result for JSON serialization"""

        if isinstance(result, dict):
            sanitized = {}
            for k, v in result.items():
                if isinstance(v, (str, int, float, bool)):
                    sanitized[k] = v
                elif isinstance(v, np.ndarray):
                    sanitized[k] = v.tolist() if v.size < 10 else f"array_shape_{v.shape}"
                elif hasattr(v, '__dict__'):
                    sanitized[k] = str(v)
                else:
                    sanitized[k] = str(v)
            return sanitized

        return {'result': str(result)}

    def _calculate_agent_state_change(self, pre_state: Dict, post_state: Dict) -> Dict[str, Any]:
        """Calculate the magnitude of agent state changes"""

        changes = {}

        # Goal change
        changes['goal_changed'] = pre_state.get('goal') != post_state.get('goal')

        # Context change
        changes['context_changed'] = pre_state.get('context') != post_state.get('context')

        # Q-value changes
        pre_q = pre_state.get('q_values', {})
        post_q = post_state.get('q_values', {})
        q_change = 0.0
        for goal in set(pre_q.keys()) | set(post_q.keys()):
            q_change += abs(post_q.get(goal, 0) - pre_q.get(goal, 0))
        changes['q_value_change'] = q_change

        # Memory changes
        changes['memory_working_change'] = post_state.get('memory_working_size', 0) - pre_state.get('memory_working_size', 0)
        changes['memory_episodic_change'] = post_state.get('memory_episodic_size', 0) - pre_state.get('memory_episodic_size', 0)

        # Embodied changes
        if 'position' in pre_state and 'position' in post_state and pre_state['position'] is not None and post_state['position'] is not None:
            pos_change = np.linalg.norm(np.array(post_state['position']) - np.array(pre_state['position']))
            changes['position_change'] = float(pos_change)
            changes['energy_change'] = post_state.get('energy', 0) - pre_state.get('energy', 0)

        return changes

    def _detect_decision_event(self, pre_state: Dict, post_state: Dict, qse_metrics: Dict) -> bool:
        """Detect if a significant decision event occurred"""

        state_changes = self._calculate_agent_state_change(pre_state, post_state)

        # Decision indicators
        decision_indicators = [
            state_changes.get('goal_changed', False),
            state_changes.get('context_changed', False),
            state_changes.get('q_value_change', 0) > 0.1,
            qse_metrics.get('phasic_rupture_active', False)
        ]

        return any(decision_indicators)

    def _calculate_qse_influence(self, qse_metrics: Dict, pre_state: Dict, post_state: Dict) -> float:
        """Calculate how much QSE dynamics influenced agent behavior"""

        # Normalized influence score based on QSE activity and agent changes
        influence_factors = []

        # Tau influence (emergent time pressure)
        tau_util = qse_metrics.get('tau_utilization', 0.5)
        influence_factors.append(abs(tau_util - 0.5) * 2)  # Distance from neutral

        # Sigma influence (symbolic pressure)
        sigma_abs = qse_metrics.get('sigma_abs_mean', 0)
        influence_factors.append(min(sigma_abs, 1.0))

        # Rupture influence (instability pressure)
        tonic_rupture = 1.0 if qse_metrics.get('tonic_rupture_active', False) else 0.0
        phasic_rupture = 2.0 if qse_metrics.get('phasic_rupture_active', False) else 0.0
        influence_factors.append((tonic_rupture + phasic_rupture) / 3.0)

        # Entropy influence (information pressure)
        entropy = qse_metrics.get('prob_entropy_normalized', 0.5)
        influence_factors.append(abs(entropy - 0.8))  # Distance from healthy high entropy

        # Weight by actual agent changes
        state_changes = self._calculate_agent_state_change(pre_state, post_state)
        agent_activity = sum([
            1.0 if state_changes.get('goal_changed', False) else 0.0,
            1.0 if state_changes.get('context_changed', False) else 0.0,
            min(state_changes.get('q_value_change', 0), 1.0),
            min(state_changes.get('position_change', 0), 1.0) if 'position_change' in state_changes else 0.0
        ]) / 4.0

        # Combined influence score
        qse_pressure = np.mean(influence_factors)
        influence_score = qse_pressure * agent_activity

        return float(np.clip(influence_score, 0.0, 1.0))

    def _analyze_dynamics_patterns(self):
        """Analyze patterns in QSE dynamics and agent behavior"""

        if not self.dynamics_history:
            return

        # Extract time series
        steps = len(self.dynamics_history)

        # QSE dynamics patterns
        tau_series = [d['tau_current'] for d in self.dynamics_history]
        sigma_series = [d['sigma_mean'] for d in self.dynamics_history]
        tonic_rupture_series = [d['tonic_rupture_active'] for d in self.dynamics_history]
        phasic_rupture_series = [d['phasic_rupture_active'] for d in self.dynamics_history]
        entropy_series = [d['prob_entropy_normalized'] for d in self.dynamics_history]

        # Agent behavior patterns
        goal_changes = [d['agent_state_change']['goal_changed'] for d in self.dynamics_history]
        context_changes = [d['agent_state_change']['context_changed'] for d in self.dynamics_history]
        q_value_changes = [d['agent_state_change']['q_value_change'] for d in self.dynamics_history]
        influence_scores = [d['qse_influence_score'] for d in self.dynamics_history]

        # Pattern analysis
        self.dynamics_patterns = {
            'qse_dynamics': {
                'tau_mean': float(np.mean(tau_series)),
                'tau_std': float(np.std(tau_series)),
                'sigma_mean': float(np.mean(sigma_series)),
                'sigma_std': float(np.std(sigma_series)),
                'tonic_rupture_fraction': float(np.mean(tonic_rupture_series)),
                'phasic_rupture_fraction': float(np.mean(phasic_rupture_series)),
                'entropy_mean': float(np.mean(entropy_series)),
                'entropy_std': float(np.std(entropy_series))
            },

            'agent_behavior': {
                'goal_change_rate': float(np.mean(goal_changes)),
                'context_change_rate': float(np.mean(context_changes)),
                'q_value_change_mean': float(np.mean(q_value_changes)),
                'q_value_change_std': float(np.std(q_value_changes)),
                'avg_qse_influence': float(np.mean(influence_scores)),
                'max_qse_influence': float(np.max(influence_scores))
            },

            'temporal_structure': {
                'total_steps': steps,
                'decision_events': int(np.sum([d['decision_triggered'] for d in self.dynamics_history])),
                'high_influence_periods': int(np.sum([s > 0.5 for s in influence_scores]))
            }
        }

    def _analyze_dynamics_correlations(self):
        """Analyze correlations between QSE dynamics and agent behavior"""

        if len(self.dynamics_history) < 10:
            return

        # Create correlation matrix
        correlation_data = []

        for d in self.dynamics_history:
            correlation_data.append({
                # QSE variables
                'tau_current': d['tau_current'],
                'sigma_mean': d['sigma_mean'],
                'tonic_rupture_active': int(d['tonic_rupture_active']),
                'phasic_rupture_active': int(d['phasic_rupture_active']),
                'prob_entropy_normalized': d['prob_entropy_normalized'],

                # Agent variables
                'goal_changed': int(d['agent_state_change']['goal_changed']),
                'context_changed': int(d['agent_state_change']['context_changed']),
                'q_value_change': d['agent_state_change']['q_value_change'],
                'qse_influence_score': d['qse_influence_score']
            })

        df = pd.DataFrame(correlation_data)

        # Compute correlation matrices
        self.correlation_matrices['pearson'] = df.corr(method='pearson')
        self.correlation_matrices['spearman'] = df.corr(method='spearman')

        # Compute lagged correlations (QSE leads behavior)
        self._compute_lagged_correlations(df)

    def _compute_lagged_correlations(self, df: pd.DataFrame, max_lag: int = 10):
        """Compute time-lagged correlations to identify QSE → behavior causality"""

        qse_vars = ['tau_current', 'sigma_mean', 'phasic_rupture_active', 'prob_entropy_normalized']
        behavior_vars = ['goal_changed', 'context_changed', 'q_value_change']

        self.correlation_matrices['lagged'] = {}

        for qse_var in qse_vars:
            for behavior_var in behavior_vars:
                lagged_corrs = []

                for lag in range(max_lag + 1):
                    if lag == 0:
                        corr = df[qse_var].corr(df[behavior_var])
                    else:
                        if len(df) > lag:
                            corr = df[qse_var][:-lag].corr(df[behavior_var][lag:])
                        else:
                            corr = np.nan

                    lagged_corrs.append(corr)

                self.correlation_matrices['lagged'][f"{qse_var}_leads_{behavior_var}"] = lagged_corrs

    def _identify_causal_relationships(self):
        """Identify causal relationships between QSE dynamics and agent behavior"""

        causal_relationships = {}

        if 'lagged' in self.correlation_matrices:
            for relationship, lagged_corrs in self.correlation_matrices['lagged'].items():

                # Check if we have any valid correlations
                valid_corrs = [c for c in lagged_corrs if not np.isnan(c)]
                if not valid_corrs:
                    continue  # Skip if all correlations are NaN
                
                # Find peak correlation and its lag, handling NaN values
                abs_corrs = np.abs(lagged_corrs)
                if np.all(np.isnan(abs_corrs)):
                    continue  # Skip if all are NaN
                    
                max_corr_idx = np.nanargmax(abs_corrs)
                max_corr = lagged_corrs[max_corr_idx]

                if not np.isnan(max_corr) and abs(max_corr) > 0.2 and max_corr_idx > 0:  # Significant lagged correlation
                    qse_var, behavior_var = relationship.split('_leads_')

                    causal_relationships[relationship] = {
                        'qse_variable': qse_var,
                        'behavior_variable': behavior_var,
                        'optimal_lag': int(max_corr_idx),
                        'strength': float(max_corr),
                        'significant': abs(max_corr) > 0.3,
                        'interpretation': self._interpret_causal_relationship(qse_var, behavior_var, max_corr_idx, max_corr)
                    }

        self.causal_relationships = causal_relationships

    def _interpret_causal_relationship(self, qse_var: str, behavior_var: str, lag: int, strength: float) -> str:
        """Provide interpretation of causal relationships"""

        direction = "increases" if strength > 0 else "decreases"

        interpretations = {
            ('tau_current', 'goal_changed'): f"Emergent time changes {direction} goal switching after {lag} steps",
            ('sigma_mean', 'context_changed'): f"Symbolic curvature {direction} context shifts after {lag} steps",
            ('phasic_rupture_active', 'goal_changed'): f"Regime transitions {direction} goal changes after {lag} steps",
            ('prob_entropy_normalized', 'q_value_change'): f"Quantum entropy {direction} learning after {lag} steps"
        }

        return interpretations.get((qse_var, behavior_var),
                                f"{qse_var} {direction} {behavior_var} with {lag}-step delay")

    def _generate_dynamics_report(self) -> Dict[str, Any]:
        """Generate comprehensive dynamics-behavior report"""

        report = {
            'experiment_metadata': {
                'agent_type': self.agent_type,
                'environment_type': self.environment_type,
                'total_steps': len(self.dynamics_history),
                'analysis_timestamp': time.time()
            },

            'dynamics_patterns': self.dynamics_patterns,
            'correlation_matrices': {k: v.to_dict() if hasattr(v, 'to_dict') else v
                                   for k, v in self.correlation_matrices.items()},
            'causal_relationships': self.causal_relationships,

            'key_findings': self._extract_key_dynamics_findings()
        }

        return report

    def _extract_key_dynamics_findings(self) -> Dict[str, Any]:
        """Extract key findings about QSE-agent dynamics"""

        findings = {}

        if self.dynamics_patterns:
            # Autopoietic health
            tonic_fraction = self.dynamics_patterns['qse_dynamics']['tonic_rupture_fraction']
            phasic_fraction = self.dynamics_patterns['qse_dynamics']['phasic_rupture_fraction']

            findings['autopoietic_health'] = {
                'tonic_aliveness': 'excellent' if tonic_fraction > 0.9 else 'good' if tonic_fraction > 0.7 else 'poor',
                'regime_dynamics': 'excellent' if 0.1 <= phasic_fraction <= 0.4 else 'moderate',
                'overall_assessment': 'healthy' if tonic_fraction > 0.8 and 0.05 <= phasic_fraction <= 0.5 else 'concerning'
            }

            # Agent responsiveness
            avg_influence = self.dynamics_patterns['agent_behavior']['avg_qse_influence']
            decision_rate = self.dynamics_patterns['temporal_structure']['decision_events'] / self.dynamics_patterns['temporal_structure']['total_steps']

            findings['agent_responsiveness'] = {
                'qse_influence_level': 'high' if avg_influence > 0.3 else 'moderate' if avg_influence > 0.1 else 'low',
                'decision_frequency': 'high' if decision_rate > 0.1 else 'moderate' if decision_rate > 0.05 else 'low',
                'coupling_strength': 'strong' if avg_influence > 0.2 and decision_rate > 0.08 else 'weak'
            }

        # Causal insights
        if self.causal_relationships:
            strongest_causal = max(self.causal_relationships.items(),
                                 key=lambda x: abs(x[1]['strength']))

            findings['causal_insights'] = {
                'strongest_relationship': strongest_causal[0],
                'strength': strongest_causal[1]['strength'],
                'lag': strongest_causal[1]['optimal_lag'],
                'interpretation': strongest_causal[1]['interpretation'],
                'total_relationships': len(self.causal_relationships)
            }
        else:
            findings['causal_insights'] = {
                'strongest_relationship': None,
                'strength': 0.0,
                'lag': 0,
                'interpretation': 'No significant causal relationships detected between QSE dynamics and agent behavior',
                'total_relationships': 0
            }

        return findings

    def _create_dynamics_visualizations(self):
        """Create comprehensive dynamics-behavior visualizations"""

        if len(self.dynamics_history) < 10:
            return

        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        fig.suptitle(f'QSE-Agent Dynamics Analysis ({self.agent_type} agent)', fontsize=16, fontweight='bold')

        # Extract time series
        steps = range(len(self.dynamics_history))
        tau_series = [d['tau_current'] for d in self.dynamics_history]
        sigma_series = [d['sigma_mean'] for d in self.dynamics_history]
        influence_series = [d['qse_influence_score'] for d in self.dynamics_history]

        # 1. QSE Dynamics Overview
        ax1 = axes[0, 0]
        ax1.plot(steps, tau_series, label='τ (emergent time)', alpha=0.7)
        ax1.plot(steps, sigma_series, label='σ (curvature)', alpha=0.7)
        ax1.set_title('QSE Dynamics')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. QSE Influence on Agent
        ax2 = axes[0, 1]
        ax2.plot(steps, influence_series, 'purple', linewidth=2)
        ax2.fill_between(steps, influence_series, alpha=0.3, color='purple')
        ax2.set_title('QSE Influence on Agent')
        ax2.set_ylabel('Influence Score')
        ax2.grid(True, alpha=0.3)

        # 3. Agent Decision Events
        ax3 = axes[0, 2]
        decision_events = [i for i, d in enumerate(self.dynamics_history) if d['decision_triggered']]
        if decision_events:
            ax3.scatter(decision_events, [1]*len(decision_events), alpha=0.6, s=20)
            ax3.set_ylim(0, 2)
        ax3.set_title(f'Decision Events ({len(decision_events)} total)')
        ax3.grid(True, alpha=0.3)

        # 4. Correlation Heatmap
        if 'pearson' in self.correlation_matrices:
            ax4 = axes[1, 0]
            corr_matrix = self.correlation_matrices['pearson']
            sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, ax=ax4,
                       cbar_kws={'label': 'Correlation'})
            ax4.set_title('QSE-Agent Correlations')

        # 5. Causal Relationships
        if self.causal_relationships:
            ax5 = axes[1, 1]
            relationships = list(self.causal_relationships.keys())[:8]  # Top 8
            strengths = [self.causal_relationships[r]['strength'] for r in relationships]

            bars = ax5.barh(range(len(relationships)), strengths)
            ax5.set_yticks(range(len(relationships)))
            ax5.set_yticklabels([r.replace('_leads_', ' → ') for r in relationships])
            ax5.set_title('Causal Relationships')
            ax5.set_xlabel('Correlation Strength')

        # 6. Agent State Evolution
        ax6 = axes[1, 2]
        goal_changes = [i for i, d in enumerate(self.dynamics_history) if d['agent_state_change']['goal_changed']]
        context_changes = [i for i, d in enumerate(self.dynamics_history) if d['agent_state_change']['context_changed']]

        if goal_changes:
            ax6.scatter(goal_changes, [1]*len(goal_changes), label='Goal Changes', alpha=0.6, s=30)
        if context_changes:
            ax6.scatter(context_changes, [2]*len(context_changes), label='Context Changes', alpha=0.6, s=30)

        ax6.set_ylim(0, 3)
        ax6.set_title('Agent State Changes')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # 7-9. Time series subplots
        if len(self.dynamics_history) > 100:
            # Show a representative section
            start_idx = len(self.dynamics_history) // 4
            end_idx = start_idx + 200

            section_steps = range(start_idx, min(end_idx, len(self.dynamics_history)))
            section_tau = tau_series[start_idx:end_idx]
            section_influence = influence_series[start_idx:end_idx]

            ax7 = axes[2, 0]
            ax7.plot(section_steps, section_tau, 'blue', linewidth=2, label='τ')
            ax7_twin = ax7.twinx()
            ax7_twin.plot(section_steps, section_influence, 'red', linewidth=2, label='Influence')
            ax7.set_title('Detailed Time Evolution')
            ax7.legend(loc='upper left')
            ax7_twin.legend(loc='upper right')

            # Mark decision events in this section
            section_decisions = [i for i in decision_events if start_idx <= i < end_idx]
            if section_decisions:
                decision_tau = [tau_series[i] for i in section_decisions]
                ax7.scatter(section_decisions, decision_tau, color='green', s=50, marker='^',
                           label='Decisions', zorder=5)

        plt.tight_layout()
        plt.savefig('qse_agent_dynamics_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        print(f"  ✅ Visualization saved as 'qse_agent_dynamics_analysis.png'")


def main():
    parser = argparse.ArgumentParser(description='QSE Agent Dynamics Analysis')
    parser.add_argument('--steps', type=int, default=10000, help='Number of steps to run')
    parser.add_argument('--agent', choices=['cognitive', 'embodied'], default='cognitive',
                       help='Type of agent to analyze')
    parser.add_argument('--environment', choices=['basic', 'embodied'], default='basic',
                       help='Type of environment')
    parser.add_argument('--dt', type=float, default=0.01, help='Time step')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='qse_agent_dynamics.jsonl', help='Output file')
    parser.add_argument('--analysis', choices=['basic', 'deep'], default='deep',
                       help='Analysis depth')

    args = parser.parse_args()

    # Auto-select environment based on agent type
    if args.agent == 'embodied' and args.environment == 'basic':
        args.environment = 'embodied'

    # Run analysis
    runner = QSEAgentDynamicsRunner(agent_type=args.agent, environment_type=args.environment)
    results = runner.run_dynamics_analysis(
        steps=args.steps,
        dt=args.dt,
        output_file=args.output,
        seed=args.seed
    )

    # Print key findings
    print(f"\n🔬 KEY DYNAMICS-BEHAVIOR FINDINGS:")
    print(f"=" * 60)

    findings = results.get('key_findings', {})

    if 'autopoietic_health' in findings:
        health = findings['autopoietic_health']
        print(f"🧬 Autopoietic Health: {health['overall_assessment']}")
        print(f"   Tonic aliveness: {health['tonic_aliveness']}")
        print(f"   Regime dynamics: {health['regime_dynamics']}")

    if 'agent_responsiveness' in findings:
        resp = findings['agent_responsiveness']
        print(f"🤖 Agent Responsiveness: {resp['coupling_strength']} coupling")
        print(f"   QSE influence: {resp['qse_influence_level']}")
        print(f"   Decision frequency: {resp['decision_frequency']}")

    if 'causal_insights' in findings:
        causal = findings['causal_insights']
        print(f"🔗 Strongest Causal Relationship:")
        print(f"   {causal['interpretation']}")
        print(f"   Strength: {causal['strength']:.3f}, Lag: {causal['lag']} steps")
        print(f"   Total relationships found: {causal['total_relationships']}")

    print(f"\n📁 Full results saved to: {args.output}")
    print(f"📊 Visualization: qse_agent_dynamics_analysis.png")


if __name__ == "__main__":
    main()
