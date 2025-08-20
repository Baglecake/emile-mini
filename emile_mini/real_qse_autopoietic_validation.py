
#!/usr/bin/env python3
"""
Real QSE Autopoietic Validation Suite
=====================================

Validates computational autopoiesis using REAL QSE system dynamics:
- Generates authentic quantum-embodied-social dynamics
- Uses sophisticated QSE dynamics analyzer for rich pattern discovery
- Extracts genuine autopoietic evidence from emergent properties
- Compares multi-modal autopoietic manifestations

Key Features:
1. Real quantum dynamics with Schrödinger evolution
2. Embodied sensorimotor-QSE integration  
3. Social multi-agent QSE resonance
4. Advanced temporal pattern analysis
5. Genuine autopoietic evidence extraction

Usage:
    python real_qse_autopoietic_validation.py --comprehensive
    python real_qse_autopoietic_validation.py --modality embodied
    python real_qse_autopoietic_validation.py --compare-modalities
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
import subprocess
import tempfile
import os
warnings.filterwarnings('ignore')

# Import real QSE components
from emile_mini.config import QSEConfig
from emile_mini.agent import EmileAgent
from emile_mini.embodied_qse_emile import EmbodiedQSEAgent, EmbodiedEnvironment, run_embodied_experiment
from emile_mini.social_qse_agent_v2 import SocialQSEAgent, SocialEnvironment, run_social_experiment

try:
    from analyze_qse_dynamics import QSEDynamicsAnalyzer
    QSE_ANALYZER_AVAILABLE = True
except ImportError:
    QSE_ANALYZER_AVAILABLE = False
    print("⚠️ analyze_qse_dynamics.py not found - core functionality disabled")


class RealQSEAutopoieticValidator:
    """Validates autopoiesis using real QSE system dynamics"""
    
    def __init__(self, output_dir: str = "real_qse_autopoiesis_validation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results = {}
        self.qse_data_files = {}
        self.qse_findings = {}
        
        print(f"🧬 REAL QSE AUTOPOIETIC VALIDATION SUITE")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🔬 Using authentic quantum-embodied-social dynamics")
    
    def generate_basic_qse_dynamics(self, steps: int = 15000) -> str:
        """Generate basic QSE agent dynamics"""
        print(f"\n🔧 Generating basic QSE dynamics ({steps} steps)...")
        
        output_file = self.output_dir / "basic_qse_dynamics.jsonl"
        
        # Create basic QSE agent
        cfg = QSEConfig()
        agent = EmileAgent(cfg)
        
        # Add goals
        for goal in ["explore", "exploit", "maintain", "adapt", "learn"]:
            agent.goal.add_goal(goal)
        
        # Generate dynamics with rich logging
        with open(output_file, 'w') as f:
            for step in range(steps):
                if step % 1000 == 0:
                    print(f"    Step {step}/{steps}")
                
                # Get pre-step state
                pre_state = {
                    'context': agent.context.get_current(),
                    'goal': agent.goal.current_goal,
                    'q_values': getattr(agent.goal, 'q_values', {}).copy()
                }
                
                # Agent step
                metrics = agent.step()
                
                # Get post-step state
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
                
                # Extract rich QSE metrics
                tau_current = getattr(agent.qse, 'tau', metrics.get('emergent_time', 0.5))
                sigma_mean = metrics.get('sigma_mean', getattr(agent.symbolic, 'sigma_ema', 0))
                surplus_mean = metrics.get('surplus_mean', np.mean(agent.qse.S))
                
                # Detect regime states
                phasic_rupture = abs(sigma_mean) > 0.7
                tonic_rupture = abs(sigma_mean) > 0.3
                
                # QSE influence calculation
                qse_influence = abs(surplus_mean) * 0.3 + abs(sigma_mean) * 0.4 + abs(tau_current - 0.5) * 0.3
                
                # Create rich data entry
                data_entry = {
                    'step': step,
                    'tau_current': float(tau_current),
                    'sigma_mean': float(sigma_mean),
                    'surplus_mean': float(surplus_mean),
                    'tonic_rupture_active': bool(tonic_rupture),
                    'phasic_rupture_active': bool(phasic_rupture),
                    'prob_entropy_normalized': float(metrics.get('normalized_entropy', 0.5)),
                    'agent_pre_state': pre_state,
                    'agent_post_state': post_state,
                    'agent_state_change': state_changes,
                    'decision_triggered': bool(state_changes['goal_changed'] or state_changes['context_changed']),
                    'qse_influence_score': float(qse_influence),
                    'modality': 'basic'
                }
                
                # Write to JSONL
                f.write(json.dumps(data_entry) + '\n')
        
        print(f"✅ Basic QSE dynamics saved to: {output_file}")
        return str(output_file)
    
    def generate_embodied_qse_dynamics(self, steps: int = 15000) -> str:
        """Generate embodied QSE dynamics with sensorimotor integration"""
        print(f"\n🤖 Generating embodied QSE dynamics ({steps} steps)...")
        
        output_file = self.output_dir / "embodied_qse_dynamics.jsonl"
        
        # Create embodied environment and agent
        env = EmbodiedEnvironment(size=20)
        agent = EmbodiedQSEAgent()
        
        # Track rich embodied data
        with open(output_file, 'w') as f:
            for step in range(steps):
                if step % 1000 == 0:
                    print(f"    Step {step}/{steps}")
                
                # Get detailed pre-step state
                pre_state = {
                    'context': agent.context.get_current(),
                    'goal': agent.goal.current_goal,
                    'position': agent.body.state.position,
                    'energy': agent.body.state.energy,
                    'orientation': agent.body.state.orientation
                }
                
                # Execute embodied step
                result = agent.embodied_step(env)
                
                # Get post-step state
                post_state = {
                    'context': result['context'],
                    'goal': agent.goal.current_goal,
                    'position': agent.body.state.position,
                    'energy': agent.body.state.energy,
                    'orientation': agent.body.state.orientation
                }
                
                # Detect changes
                state_changes = {
                    'goal_changed': pre_state['goal'] != post_state['goal'],
                    'context_changed': result['context_switched'],
                    'position_changed': pre_state['position'] != post_state['position'],
                    'q_value_change': 0.1 if result['reward'] != 0 else 0  # Simplified
                }
                
                # Extract embodied QSE metrics
                cognitive_metrics = result['cognitive_metrics']
                tau_current = getattr(agent.qse, 'tau', cognitive_metrics.get('emergent_time', 0.5))
                sigma_mean = cognitive_metrics.get('sigma_mean', 0)
                surplus_mean = cognitive_metrics.get('surplus_mean', 0)
                
                # Embodied-specific ruptures (sensorimotor-driven)
                phasic_rupture = (abs(sigma_mean) > 0.6 or 
                                result['environment_feedback'].get('collision_type') is not None)
                tonic_rupture = abs(sigma_mean) > 0.25
                
                # Embodied QSE influence (includes spatial and energetic factors)
                spatial_change = 1.0 if state_changes['position_changed'] else 0.0
                energy_influence = abs(post_state['energy'] - pre_state['energy']) * 2.0
                qse_influence = (abs(surplus_mean) * 0.2 + abs(sigma_mean) * 0.3 + 
                               abs(tau_current - 0.5) * 0.2 + spatial_change * 0.2 + energy_influence * 0.1)
                
                # Rich embodied data entry
                data_entry = {
                    'step': step,
                    'tau_current': float(tau_current),
                    'sigma_mean': float(sigma_mean),
                    'surplus_mean': float(surplus_mean),
                    'tonic_rupture_active': bool(tonic_rupture),
                    'phasic_rupture_active': bool(phasic_rupture),
                    'prob_entropy_normalized': float(cognitive_metrics.get('normalized_entropy', 0.5)),
                    'agent_pre_state': pre_state,
                    'agent_post_state': post_state,
                    'agent_state_change': state_changes,
                    'decision_triggered': bool(state_changes['goal_changed'] or state_changes['context_changed']),
                    'qse_influence_score': float(qse_influence),
                    'modality': 'embodied',
                    'embodied_metrics': {
                        'action': result['action'],
                        'reward': result['reward'],
                        'body_energy': post_state['energy'],
                        'spatial_change': spatial_change,
                        'environment_outcome': result['environment_feedback'].get('outcome', 'none')
                    }
                }
                
                f.write(json.dumps(data_entry) + '\n')
        
        print(f"✅ Embodied QSE dynamics saved to: {output_file}")
        return str(output_file)
    
    def generate_social_qse_dynamics(self, steps: int = 8000, n_agents: int = 3) -> str:
        """Generate social QSE dynamics with multi-agent interactions"""
        print(f"\n👥 Generating social QSE dynamics ({n_agents} agents, {steps} steps)...")
        
        output_file = self.output_dir / "social_qse_dynamics.jsonl"
        
        # Create social environment and agents
        env = SocialEnvironment(size=15)
        agents = []
        
        for i in range(n_agents):
            agent = SocialQSEAgent(f"Agent_{i}")
            env.add_agent(agent)
            agents.append(agent)
        
        # Cluster spawn for social interaction
        cx, cy = env.size // 2, env.size // 2
        agents[0].body.state.position = (cx, cy)
        for i, agent in enumerate(agents[1:], 1):
            dx, dy = np.random.randint(-2, 3, 2)
            agent.body.state.position = (cx + dx, cy + dy)
        
        # Prime expert agent with knowledge
        if agents:
            expert = agents[0]
            expert.embodied_mappings['crimson_fruit'].extend([-0.9] * 3)
            expert.embodied_mappings['blue_fruit'].extend([0.3] * 2)
        
        # Generate social dynamics
        with open(output_file, 'w') as f:
            for step in range(steps):
                if step % 500 == 0:
                    print(f"    Step {step}/{steps}")
                
                # Step all agents and collect rich social data
                step_results = env.step_all_agents()
                
                # Create entries for each agent
                for agent_id, agent in env.agents.items():
                    result = step_results.get(agent_id, {})
                    
                    # Get social state
                    social_info = result.get('social_info', {})
                    
                    # Extract QSE metrics (with social modulation)
                    cognitive_metrics = result.get('cognitive_metrics', {})
                    tau_current = getattr(agent.qse, 'tau', 0.5)
                    sigma_mean = cognitive_metrics.get('sigma_mean', 0)
                    surplus_mean = cognitive_metrics.get('surplus_mean', 0)
                    
                    # Social-specific ruptures
                    social_signals = len(agent.outgoing_signals) + len(agent.incoming_signals)
                    teaching_active = agent.current_social_strategy == 'teaching'
                    phasic_rupture = (abs(sigma_mean) > 0.5 or social_signals > 2 or teaching_active)
                    tonic_rupture = abs(sigma_mean) > 0.2
                    
                    # Social QSE influence (includes social interaction effects)
                    social_influence = min(1.0, social_signals * 0.1)
                    knowledge_influence = len(agent.social_knowledge_transfer) * 0.05
                    proximity_influence = min(1.0, social_info.get('nearby_agents', 0) * 0.2)
                    
                    qse_influence = (abs(surplus_mean) * 0.2 + abs(sigma_mean) * 0.3 + 
                                   abs(tau_current - 0.5) * 0.2 + social_influence * 0.2 + 
                                   knowledge_influence * 0.05 + proximity_influence * 0.05)
                    
                    # Detect state changes
                    state_changes = {
                        'goal_changed': result.get('goal_changed', False),
                        'context_changed': result.get('context_switched', False),
                        'social_strategy_changed': hasattr(agent, '_previous_strategy') and 
                                                 agent.current_social_strategy != getattr(agent, '_previous_strategy', ''),
                        'q_value_change': 0.1 if result.get('reward', 0) != 0 else 0
                    }
                    
                    # Social data entry
                    data_entry = {
                        'step': step,
                        'agent_id': agent_id,
                        'tau_current': float(tau_current),
                        'sigma_mean': float(sigma_mean),
                        'surplus_mean': float(surplus_mean),
                        'tonic_rupture_active': bool(tonic_rupture),
                        'phasic_rupture_active': bool(phasic_rupture),
                        'prob_entropy_normalized': float(cognitive_metrics.get('normalized_entropy', 0.5)),
                        'agent_pre_state': {'context': agent.context.get_current(), 'goal': agent.goal.current_goal},
                        'agent_post_state': {'context': agent.context.get_current(), 'goal': agent.goal.current_goal},
                        'agent_state_change': state_changes,
                        'decision_triggered': bool(state_changes['goal_changed'] or state_changes['context_changed']),
                        'qse_influence_score': float(qse_influence),
                        'modality': 'social',
                        'social_metrics': {
                            'strategy': agent.current_social_strategy,
                            'nearby_agents': social_info.get('nearby_agents', 0),
                            'social_signals': social_signals,
                            'knowledge_transfers': len(agent.social_knowledge_transfer),
                            'personality': dict(agent.social_personality),
                            'social_action': social_info.get('social_action', 'none')
                        }
                    }
                    
                    f.write(json.dumps(data_entry) + '\n')
                    
                    # Store previous strategy for change detection
                    agent._previous_strategy = agent.current_social_strategy
        
        print(f"✅ Social QSE dynamics saved to: {output_file}")
        return str(output_file)
    
    def analyze_qse_dynamics(self, jsonl_file: str, modality: str) -> Dict[str, Any]:
        """Analyze QSE dynamics using the sophisticated analyzer"""
        print(f"\n🔬 Analyzing {modality} QSE dynamics...")
        
        if not QSE_ANALYZER_AVAILABLE:
            print("⚠️ QSE dynamics analyzer not available - using simplified analysis")
            return self._simplified_analysis(jsonl_file, modality)
        
        try:
            # Use the real QSE dynamics analyzer
            analyzer = QSEDynamicsAnalyzer([jsonl_file])
            findings = analyzer.run_complete_analysis(deep=True)
            
            print(f"✅ {modality.title()} QSE analysis complete")
            print(f"📊 Found {len(findings.get('decision_chains', []))} decision chains")
            print(f"🔄 Found {len(findings.get('regime_transitions', []))} regime transitions")
            
            return findings
            
        except Exception as e:
            print(f"⚠️ QSE analyzer failed: {e}")
            return self._simplified_analysis(jsonl_file, modality)
    
    def extract_autopoietic_evidence(self, findings: Dict[str, Any], modality: str) -> Dict[str, Any]:
        """Extract autopoietic evidence from QSE dynamics findings"""
        print(f"\n🧬 Extracting autopoietic evidence from {modality} dynamics...")
        
        evidence = {
            'modality': modality,
            'qse_causality': {},
            'regime_autonomy': {},
            'recursive_causality': {},
            'boundary_maintenance': {},
            'self_organization': {}
        }
        
        # 1. QSE Causality (QSE → Behavior)
        if 'advanced_correlations' in findings:
            corr_data = findings['advanced_correlations']
            
            # Extract strongest QSE-behavior correlations
            cross_corrs = corr_data.get('cross_correlations', {})
            strongest_corr = 0
            best_correlation = ('none', 0)
            
            for name, data in cross_corrs.items():
                max_corr = data.get('max_corr', 0)
                if max_corr > strongest_corr:
                    strongest_corr = max_corr
                    best_correlation = (name, max_corr)
            
            # Conditional correlations (regime-dependent causality)
            conditional = corr_data.get('conditional', {})
            
            evidence['qse_causality'] = {
                'strongest_correlation': best_correlation,
                'correlation_strength': strongest_corr,
                'phasic_decision_correlation': conditional.get('tau_vs_decisions_during_phasic', 0),
                'stable_decision_correlation': conditional.get('tau_vs_decisions_during_stable', 0),
                'regime_dependent_causality': abs(conditional.get('tau_vs_decisions_during_phasic', 0)) > 
                                            abs(conditional.get('tau_vs_decisions_during_stable', 0)),
                'causality_assessment': self._assess_causality_strength(strongest_corr, conditional)
            }
        
        # 2. Regime Autonomy (Distinct behavioral regimes)
        if 'regime_transitions' in findings:
            transitions = findings['regime_transitions']
            
            # Analyze transition quality and behavioral consequences
            behavioral_responses = []
            for transition in transitions:
                response = transition.get('behavioral_response', {})
                decision_events = response.get('decision_events', 0)
                behavioral_responses.append(decision_events)
            
            evidence['regime_autonomy'] = {
                'total_transitions': len(transitions),
                'avg_behavioral_response': np.mean(behavioral_responses) if behavioral_responses else 0,
                'transition_quality': np.std(behavioral_responses) if len(behavioral_responses) > 1 else 0,
                'regime_coherence': len(transitions) > 0,
                'autonomy_strength': min(1.0, len(transitions) / 10.0),
                'autonomy_assessment': self._assess_regime_autonomy(len(transitions), behavioral_responses)
            }
        
        # 3. Recursive Causality (Decision chains)
        if 'decision_chains' in findings:
            chains = findings['decision_chains']
            
            chain_lengths = [c['length'] for c in chains]
            max_length = max(chain_lengths) if chain_lengths else 0
            avg_length = np.mean(chain_lengths) if chain_lengths else 0
            
            evidence['recursive_causality'] = {
                'total_chains': len(chains),
                'max_chain_length': max_length,
                'avg_chain_length': avg_length,
                'recursive_depth': sum(1 for length in chain_lengths if length >= 3),
                'recursivity_strength': min(1.0, len(chains) / 5.0),
                'recursivity_assessment': self._assess_recursive_causality(len(chains), max_length, avg_length)
            }
        
        # 4. Context Effectiveness (Sense-making)
        if 'context_effectiveness' in findings:
            ctx_eff = findings['context_effectiveness']
            
            effectiveness_scores = [data['effectiveness_score'] for data in ctx_eff.values()]
            effectiveness_variance = np.var(effectiveness_scores) if effectiveness_scores else 0
            
            evidence['boundary_maintenance'] = {
                'total_contexts': len(ctx_eff),
                'effectiveness_variance': effectiveness_variance,
                'context_differentiation': effectiveness_variance > 0.01,
                'best_context_score': max(effectiveness_scores) if effectiveness_scores else 0,
                'boundary_strength': min(1.0, effectiveness_variance * 10.0),
                'boundary_assessment': self._assess_boundary_maintenance(len(ctx_eff), effectiveness_variance)
            }
        
        # 5. Self-Organization (From basic stats)
        basic_stats = findings.get('basic_stats', {})
        total_steps = basic_stats.get('total_steps', 1)
        
        decision_rate = basic_stats.get('decision_events', 0) / total_steps
        context_rate = basic_stats.get('context_switches', 0) / total_steps
        phasic_rate = basic_stats.get('phasic_rupture_events', 0) / total_steps
        
        evidence['self_organization'] = {
            'decision_rate': decision_rate,
            'context_switch_rate': context_rate,
            'phasic_rupture_rate': phasic_rate,
            'max_qse_influence': basic_stats.get('max_qse_influence', 0),
            'organization_coherence': decision_rate > 0.05 and context_rate > 0.01,
            'organization_strength': min(1.0, (decision_rate * 10 + context_rate * 50 + phasic_rate * 100) / 3),
            'organization_assessment': self._assess_self_organization(decision_rate, context_rate, phasic_rate)
        }
        
        # Overall autopoietic assessment
        evidence['overall_assessment'] = self._compute_overall_autopoiesis(evidence)
        
        return evidence
    
    def compare_modalities(self) -> Dict[str, Any]:
        """Compare autopoietic evidence across different modalities"""
        print(f"\n🔄 Comparing autopoietic evidence across modalities...")
        
        modalities = ['basic', 'embodied', 'social']
        comparison = {
            'modality_scores': {},
            'strongest_evidence': {},
            'modality_advantages': {},
            'convergent_evidence': {},
            'divergent_patterns': {}
        }
        
        # Extract scores for each modality
        for modality in modalities:
            if modality in self.results:
                evidence = self.results[modality]['autopoietic_evidence']
                overall = evidence.get('overall_assessment', {})
                comparison['modality_scores'][modality] = overall.get('overall_autopoiesis_score', 0)
        
        # Find strongest evidence types
        evidence_types = ['qse_causality', 'regime_autonomy', 'recursive_causality', 
                         'boundary_maintenance', 'self_organization']
        
        for evidence_type in evidence_types:
            best_modality = None
            best_score = 0
            
            for modality in modalities:
                if modality in self.results:
                    evidence = self.results[modality]['autopoietic_evidence']
                    type_data = evidence.get(evidence_type, {})
                    
                    # Get strength score for this evidence type
                    strength_key = f"{evidence_type.split('_')[0]}_strength"
                    if evidence_type == 'qse_causality':
                        strength_key = 'correlation_strength'
                    elif evidence_type == 'recursive_causality':
                        strength_key = 'recursivity_strength'
                    elif evidence_type == 'boundary_maintenance':
                        strength_key = 'boundary_strength'
                    elif evidence_type == 'self_organization':
                        strength_key = 'organization_strength'
                    
                    score = type_data.get(strength_key, 0)
                    if score > best_score:
                        best_score = score
                        best_modality = modality
            
            comparison['strongest_evidence'][evidence_type] = {
                'modality': best_modality,
                'score': best_score
            }
        
        # Analyze modality-specific advantages
        comparison['modality_advantages'] = {
            'basic': "Pure QSE dynamics without embodied/social complexity",
            'embodied': "Sensorimotor integration enhances context-dependent behavior",
            'social': "Multi-agent interactions create emergent social cognition"
        }
        
        # Look for convergent evidence (similar patterns across modalities)
        convergent = {}
        for evidence_type in evidence_types:
            scores = []
            for modality in modalities:
                if modality in self.results:
                    evidence = self.results[modality]['autopoietic_evidence']
                    type_data = evidence.get(evidence_type, {})
                    strength_keys = [k for k in type_data.keys() if 'strength' in k or 'correlation' in k]
                    if strength_keys:
                        scores.append(type_data[strength_keys[0]])
            
            if len(scores) >= 2:
                convergent[evidence_type] = {
                    'scores': scores,
                    'variance': np.var(scores),
                    'convergent': np.var(scores) < 0.1  # Low variance = convergent
                }
        
        comparison['convergent_evidence'] = convergent
        
        # Overall comparison assessment
        comparison['assessment'] = self._generate_modality_comparison_assessment(comparison)
        
        return comparison
    
    def run_comprehensive_validation(self, steps_per_modality: int = 15000) -> Dict[str, Any]:
        """Run comprehensive validation across all QSE modalities"""
        
        print(f"🧬 COMPREHENSIVE REAL QSE AUTOPOIETIC VALIDATION")
        print(f"=" * 70)
        print(f"Testing authentic quantum-embodied-social autopoietic dynamics")
        
        start_time = time.time()
        
        # 1. Generate basic QSE dynamics
        basic_file = self.generate_basic_qse_dynamics(steps_per_modality)
        self.qse_data_files['basic'] = basic_file
        
        # 2. Generate embodied QSE dynamics
        embodied_file = self.generate_embodied_qse_dynamics(steps_per_modality)
        self.qse_data_files['embodied'] = embodied_file
        
        # 3. Generate social QSE dynamics
        social_file = self.generate_social_qse_dynamics(steps_per_modality // 2, n_agents=3)
        self.qse_data_files['social'] = social_file
        
        # 4. Analyze each modality
        for modality, data_file in self.qse_data_files.items():
            print(f"\n🔬 Analyzing {modality} modality...")
            
            # Deep QSE analysis
            findings = self.analyze_qse_dynamics(data_file, modality)
            self.qse_findings[modality] = findings
            
            # Extract autopoietic evidence
            evidence = self.extract_autopoietic_evidence(findings, modality)
            
            self.results[modality] = {
                'data_file': data_file,
                'qse_findings': findings,
                'autopoietic_evidence': evidence
            }
        
        # 5. Compare modalities
        comparison = self.compare_modalities()
        self.results['modality_comparison'] = comparison
        
        # 6. Overall assessment
        overall_score = self._calculate_comprehensive_autopoiesis_score()
        overall_assessment = self._generate_comprehensive_assessment(overall_score)
        publication_ready = self._assess_publication_readiness()
        
        final_results = {
            'validation_timestamp': time.time(),
            'total_runtime_minutes': (time.time() - start_time) / 60,
            'qse_data_files': self.qse_data_files,
            'modality_results': self.results,
            'modality_comparison': comparison,
            'overall_autopoiesis_score': overall_score,
            'overall_assessment': overall_assessment,
            'publication_ready': publication_ready,
            'methodology': 'real_qse_dynamics'
        }
        
        # Save results
        self._save_comprehensive_results(final_results)
        
        # Create visualizations
        self._create_comprehensive_visualizations(final_results)
        
        # Print summary
        self._print_comprehensive_summary(final_results)
        
        return final_results
    
    # Helper methods
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
    
    def _simplified_analysis(self, jsonl_file: str, modality: str) -> Dict[str, Any]:
        """Simplified analysis when full QSE analyzer is not available"""
        
        data = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except:
                    continue
        
        if not data:
            return {'basic_stats': {'total_steps': 0}}
        
        df = pd.DataFrame(data)
        
        # Basic statistics
        decision_events = df['decision_triggered'].sum()
        context_switches = df['agent_state_change'].apply(lambda x: x.get('context_changed', False)).sum()
        phasic_events = df['phasic_rupture_active'].sum()
        
        return {
            'basic_stats': {
                'total_steps': len(data),
                'decision_events': int(decision_events),
                'context_switches': int(context_switches),
                'phasic_rupture_events': int(phasic_events),
                'avg_qse_influence': float(df['qse_influence_score'].mean()),
                'max_qse_influence': float(df['qse_influence_score'].max())
            },
            'simplified': True
        }
    
    def _assess_causality_strength(self, strongest_corr: float, conditional: Dict) -> str:
        """Assess QSE causality strength"""
        if strongest_corr > 0.5:
            return "STRONG QSE causality with robust temporal patterns"
        elif strongest_corr > 0.3:
            return "MODERATE QSE causality with clear temporal relationships"
        elif strongest_corr > 0.1:
            return "WEAK QSE causality with some temporal correlations"
        else:
            return "NO significant QSE causality detected"
    
    def _assess_regime_autonomy(self, total_transitions: int, behavioral_responses: List) -> str:
        """Assess regime autonomy strength"""
        if total_transitions > 10 and np.mean(behavioral_responses) > 1.0:
            return "STRONG regime autonomy with consistent behavioral responses"
        elif total_transitions > 5:
            return "MODERATE regime autonomy with some behavioral differentiation"
        elif total_transitions > 0:
            return "WEAK regime autonomy with minimal transitions"
        else:
            return "NO regime autonomy detected"
    
    def _assess_recursive_causality(self, total_chains: int, max_length: int, avg_length: float) -> str:
        """Assess recursive causality strength"""
        if total_chains > 5 and max_length > 5:
            return "STRONG recursive causality with extended decision chains"
        elif total_chains > 2 and avg_length > 3:
            return "MODERATE recursive causality with decision cascades"
        elif total_chains > 0:
            return "WEAK recursive causality with short decision sequences"
        else:
            return "NO recursive causality detected"
    
    def _assess_boundary_maintenance(self, total_contexts: int, effectiveness_variance: float) -> str:
        """Assess boundary maintenance strength"""
        if total_contexts > 3 and effectiveness_variance > 0.05:
            return "STRONG boundary maintenance with clear context differentiation"
        elif total_contexts > 2 and effectiveness_variance > 0.02:
            return "MODERATE boundary maintenance with some context specialization"
        elif total_contexts > 1:
            return "WEAK boundary maintenance with minimal context effects"
        else:
            return "NO boundary maintenance detected"
    
    def _assess_self_organization(self, decision_rate: float, context_rate: float, phasic_rate: float) -> str:
        """Assess self-organization strength"""
        if decision_rate > 0.1 and context_rate > 0.05 and phasic_rate > 0.02:
            return "STRONG self-organization with dynamic regime management"
        elif decision_rate > 0.05 and context_rate > 0.02:
            return "MODERATE self-organization with adaptive behavior"
        elif decision_rate > 0.02:
            return "WEAK self-organization with occasional adaptation"
        else:
            return "NO self-organization detected"
    
    def _compute_overall_autopoiesis(self, evidence: Dict) -> Dict[str, Any]:
        """Compute overall autopoietic assessment for a modality"""
        
        # Extract strength scores
        causality_strength = evidence.get('qse_causality', {}).get('correlation_strength', 0)
        autonomy_strength = evidence.get('regime_autonomy', {}).get('autonomy_strength', 0)
        recursivity_strength = evidence.get('recursive_causality', {}).get('recursivity_strength', 0)
        boundary_strength = evidence.get('boundary_maintenance', {}).get('boundary_strength', 0)
        organization_strength = evidence.get('self_organization', {}).get('organization_strength', 0)
        
        # Weighted average (causality and recursivity are most important for autopoiesis)
        weights = [0.3, 0.2, 0.3, 0.1, 0.1]  # causality, autonomy, recursivity, boundary, organization
        scores = [causality_strength, autonomy_strength, recursivity_strength, boundary_strength, organization_strength]
        
        overall_score = sum(w * s for w, s in zip(weights, scores))
        
        return {
            'component_scores': {
                'qse_causality': causality_strength,
                'regime_autonomy': autonomy_strength,
                'recursive_causality': recursivity_strength,
                'boundary_maintenance': boundary_strength,
                'self_organization': organization_strength
            },
            'overall_autopoiesis_score': overall_score,
            'autopoietic_classification': (
                'STRONG' if overall_score > 0.7 else
                'MODERATE' if overall_score > 0.4 else
                'WEAK'
            ),
            'evidence_quality': 'authentic_qse_dynamics'
        }
    
    def _calculate_comprehensive_autopoiesis_score(self) -> float:
        """Calculate overall autopoiesis score across all modalities"""
        
        scores = []
        for modality in ['basic', 'embodied', 'social']:
            if modality in self.results:
                evidence = self.results[modality]['autopoietic_evidence']
                overall = evidence.get('overall_assessment', {})
                score = overall.get('overall_autopoiesis_score', 0)
                scores.append(score)
        
        if not scores:
            return 0.0
        
        # Weight embodied and social more heavily (they show richer autopoietic properties)
        if len(scores) == 3:
            weights = [0.2, 0.4, 0.4]  # basic, embodied, social
            return sum(w * s for w, s in zip(weights, scores))
        else:
            return np.mean(scores)
    
    def _generate_comprehensive_assessment(self, overall_score: float) -> str:
        """Generate comprehensive assessment"""
        if overall_score > 0.8:
            return "EXCEPTIONAL evidence for computational autopoiesis across quantum-embodied-social dynamics"
        elif overall_score > 0.65:
            return "STRONG evidence for computational autopoiesis with multi-modal validation"
        elif overall_score > 0.5:
            return "GOOD evidence for computational autopoiesis with authentic QSE dynamics"
        elif overall_score > 0.35:
            return "MODERATE evidence for autopoietic properties in real QSE system"
        else:
            return "WEAK evidence for autopoiesis - system shows primarily mechanistic behavior"
    
    def _assess_publication_readiness(self) -> bool:
        """Assess publication readiness based on evidence quality"""
        
        criteria_met = 0
        total_criteria = 5
        
        # Check for strong evidence in each modality
        for modality in ['basic', 'embodied', 'social']:
            if modality in self.results:
                evidence = self.results[modality]['autopoietic_evidence']
                overall = evidence.get('overall_assessment', {})
                score = overall.get('overall_autopoiesis_score', 0)
                
                if score > 0.6:
                    criteria_met += 1
        
        # Check for convergent evidence
        if 'modality_comparison' in self.results:
            comparison = self.results['modality_comparison']
            convergent = comparison.get('convergent_evidence', {})
            
            convergent_types = sum(1 for data in convergent.values() if data.get('convergent', False))
            if convergent_types >= 2:
                criteria_met += 1
        
        # Check overall score
        overall_score = self._calculate_comprehensive_autopoiesis_score()
        if overall_score > 0.65:
            criteria_met += 1
        
        return criteria_met >= 3
    
    def _generate_modality_comparison_assessment(self, comparison: Dict) -> str:
        """Generate assessment of modality comparison"""
        
        scores = comparison.get('modality_scores', {})
        strongest = comparison.get('strongest_evidence', {})
        
        if not scores:
            return "Insufficient data for modality comparison"
        
        best_modality = max(scores.items(), key=lambda x: x[1])
        
        assessment = f"Best overall autopoietic evidence in {best_modality[0]} modality (score: {best_modality[1]:.3f}). "
        
        # Analyze strongest evidence types
        modality_strengths = defaultdict(int)
        for evidence_type, data in strongest.items():
            modality = data.get('modality')
            if modality:
                modality_strengths[modality] += 1
        
        if modality_strengths:
            strongest_modality = max(modality_strengths.items(), key=lambda x: x[1])
            assessment += f"{strongest_modality[0].title()} shows strongest evidence in {strongest_modality[1]} categories. "
        
        # Check for convergent evidence
        convergent = comparison.get('convergent_evidence', {})
        convergent_count = sum(1 for data in convergent.values() if data.get('convergent', False))
        
        if convergent_count >= 3:
            assessment += "Strong convergent evidence across modalities supports genuine autopoietic properties."
        elif convergent_count >= 1:
            assessment += "Some convergent evidence across modalities."
        else:
            assessment += "Limited convergent evidence - modalities show different autopoietic signatures."
        
        return assessment
    
    def _save_comprehensive_results(self, results: Dict):
        """Save comprehensive results"""
        
        output_file = self.output_dir / "real_qse_autopoietic_validation_results.json"
        
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, (list, tuple, set, deque)):
                return [convert_for_json(x) for x in obj]
            if isinstance(obj, dict):
                return {str(k): convert_for_json(v) for k, v in obj.items()}
            return str(obj)
        
        json_results = convert_for_json(results)
        
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2, allow_nan=False)
        
        print(f"\n📁 Comprehensive results saved to: {output_file}")
    
    def _create_comprehensive_visualizations(self, results: Dict):
        """Create comprehensive visualizations"""
        
        # This would create rich visualizations comparing modalities
        # Implementation would depend on available data
        print(f"\n📊 Creating comprehensive visualizations...")
        
        # Placeholder for visualization creation
        # In full implementation, this would create:
        # - Modality comparison charts
        # - Autopoietic evidence radar plots
        # - QSE dynamics comparisons
        # - Temporal pattern analyses
        
        viz_file = self.output_dir / "real_qse_autopoietic_validation_summary.png"
        print(f"📊 Visualizations would be saved to: {viz_file}")
    
    def _print_comprehensive_summary(self, results: Dict):
        """Print comprehensive summary"""
        
        print(f"\n" + "=" * 70)
        print(f"🧬 REAL QSE AUTOPOIETIC VALIDATION SUMMARY")
        print(f"=" * 70)
        
        overall_score = results['overall_autopoiesis_score']
        assessment = results['overall_assessment']
        pub_ready = results['publication_ready']
        
        print(f"🔬 Overall Autopoiesis Score: {overall_score:.3f}")
        print(f"📋 Assessment: {assessment}")
        print(f"⏱️ Runtime: {results['total_runtime_minutes']:.1f} minutes")
        
        print(f"\n📊 MODALITY RESULTS:")
        modality_results = results['modality_results']
        
        for modality in ['basic', 'embodied', 'social']:
            if modality in modality_results:
                evidence = modality_results[modality]['autopoietic_evidence']
                overall = evidence.get('overall_assessment', {})
                score = overall.get('overall_autopoiesis_score', 0)
                classification = overall.get('autopoietic_classification', 'UNKNOWN')
                
                print(f"  {modality.title()}: {score:.3f} ({classification})")
                
                # Print key evidence
                for evidence_type in ['qse_causality', 'recursive_causality', 'regime_autonomy']:
                    if evidence_type in evidence:
                        assessment_key = f"{evidence_type.split('_')[0]}_assessment"
                        if evidence_type == 'qse_causality':
                            assessment_key = 'causality_assessment'
                        elif evidence_type == 'recursive_causality':
                            assessment_key = 'recursivity_assessment'
                        
                        assessment_text = evidence[evidence_type].get(assessment_key, "No assessment")
                        print(f"    - {evidence_type.replace('_', ' ').title()}: {assessment_text}")
        
        print(f"\n🎯 PUBLICATION READINESS:")
        if pub_ready:
            print(f"✅ PUBLICATION READY - Strong evidence for computational autopoiesis!")
            print(f"✅ Multi-modal validation with authentic QSE dynamics")
            print(f"✅ Quantum-embodied-social integration demonstrated")
        else:
            print(f"⚠️ Additional evidence needed for strong publication claims")
            print(f"📈 Consider longer simulation runs or parameter optimization")
        
        print(f"\n📁 Results saved to: {results['validation_timestamp']}")
        print(f"🔬 Methodology: {results['methodology']}")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Real QSE Autopoietic Validation Suite')
    parser.add_argument('--comprehensive', action='store_true', 
                       help='Run comprehensive validation across all modalities')
    parser.add_argument('--modality', choices=['basic', 'embodied', 'social'],
                       help='Run validation for specific modality')
    parser.add_argument('--compare-modalities', action='store_true',
                       help='Compare autopoietic evidence across modalities')
    parser.add_argument('--steps', type=int, default=15000, 
                       help='Steps per modality (default: 15000)')
    parser.add_argument('--output', type=str, default='real_qse_autopoiesis_validation',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = RealQSEAutopoieticValidator(output_dir=args.output)
    
    if args.comprehensive:
        # Run comprehensive validation
        results = validator.run_comprehensive_validation(steps_per_modality=args.steps)
        
    elif args.modality:
        # Run single modality
        print(f"🔬 Running {args.modality} QSE validation...")
        
        if args.modality == 'basic':
            data_file = validator.generate_basic_qse_dynamics(args.steps)
        elif args.modality == 'embodied':
            data_file = validator.generate_embodied_qse_dynamics(args.steps)
        elif args.modality == 'social':
            data_file = validator.generate_social_qse_dynamics(args.steps // 2, n_agents=3)
        
        findings = validator.analyze_qse_dynamics(data_file, args.modality)
        evidence = validator.extract_autopoietic_evidence(findings, args.modality)
        
        print(f"\n🧬 {args.modality.title()} Autopoietic Evidence:")
        overall = evidence.get('overall_assessment', {})
        score = overall.get('overall_autopoiesis_score', 0)
        classification = overall.get('autopoietic_classification', 'UNKNOWN')
        print(f"Score: {score:.3f} ({classification})")
        
    elif args.compare_modalities:
        print("🔄 Modality comparison requires comprehensive validation first")
        
    else:
        print("Please specify --comprehensive, --modality, or --compare-modalities")
        print("Example: python real_qse_autopoietic_validation.py --comprehensive")


if __name__ == "__main__":
    main()
