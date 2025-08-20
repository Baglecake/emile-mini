#!/usr/bin/env python3
"""
QSE Dynamics JSONL Analyzer - Deep Pattern Discovery
====================================================

Analyzes JSONL outputs from QSE Agent Dynamics Runner to discover:
- Temporal decision patterns and clustering
- QSE regime transitions and their behavioral consequences  
- Rare event detection and triggers
- Multi-scale correlation analysis
- Decision event sequences and chains
- Context switching patterns and effectiveness

Usage:
    python analyze_qse_dynamics.py qse_agent_dynamics.jsonl
    python analyze_qse_dynamics.py qse_agent_dynamics.jsonl --deep --export-findings
    python analyze_qse_dynamics.py multiple_files*.jsonl --compare
"""

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

def _coerce_series(x, fill=0.0):
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        arr = np.array([arr], dtype=float)
    if np.all(np.isnan(arr)):
        return np.zeros_like(arr)
    arr[np.isnan(arr)] = fill
    return arr

def _ensure_behavioral_fields(ts):
    defaults = {
        'qse_influence': 0.0,
        'q_value_change': 0.0,
        'decision_events': 0.0,
        'phasic_rupture_active': 0.0,
        'tonic_rupture_active': 0.0,
    }
    for k, v in defaults.items():
        if k not in ts:
            ts[k] = np.zeros(len(ts.get('tau', [1])))
    
    # Fix NaN/constant data
    for k in defaults.keys():
        ts[k] = _coerce_series(ts[k], fill=0.0)
    
    return ts

class QSEDynamicsAnalyzer:
    """Deep analyzer for QSE dynamics JSONL data"""
    
    def __init__(self, jsonl_files: List[str]):
        self.jsonl_files = jsonl_files
        self.data = []
        self.df = None
        self.findings = {}
        
    def _safe_corrcoef(self, x, y):
        """Safe correlation coefficient that handles NaN and constant data"""
        
        # Remove NaN values
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(x_clean) < 3:
            return 0.0
        
        # Check for constant data (no variation)
        if np.std(x_clean) < 1e-10 or np.std(y_clean) < 1e-10:
            return 0.0
        
        try:
            correlation_matrix = np.corrcoef(x_clean, y_clean)
            if correlation_matrix.shape == (2, 2):
                corr = correlation_matrix[0, 1]
                return corr if not np.isnan(corr) else 0.0
            else:
                return 0.0
        except:
            return 0.0
    
    def load_data(self):
        """Load and parse JSONL files"""
        print(f"📂 Loading {len(self.jsonl_files)} JSONL files...")
        
        all_data = []
        for file_path in self.jsonl_files:
            file_data = []
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f):
                    try:
                        entry = json.loads(line.strip())
                        entry['file_source'] = file_path
                        file_data.append(entry)
                    except json.JSONDecodeError as e:
                        print(f"⚠️ Skipping malformed line {line_num} in {file_path}")
                        continue
            
            print(f"  📄 {file_path}: {len(file_data)} steps")
            all_data.extend(file_data)
        
        self.data = all_data
        print(f"✅ Loaded total: {len(self.data)} steps")
        
        # Convert to DataFrame for analysis
        self.df = pd.DataFrame(self.data)
        return len(self.data)
    
    def extract_time_series(self) -> Dict[str, np.ndarray]:
        """FIXED VERSION - handles both QSE-only, nested agent data, and FLAT agent data"""
        
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        import numpy as np
        
        time_series = {}
        
        # QSE dynamics (always present)
        time_series['tau'] = self.df['tau_current'].values
        time_series['sigma_mean'] = self.df['sigma_mean'].values
        time_series['tonic_rupture'] = self.df['tonic_rupture_active'].astype(int).values
        time_series['phasic_rupture'] = self.df['phasic_rupture_active'].astype(int).values
        time_series['entropy'] = self.df['prob_entropy_normalized'].values
        
        # Agent behavior (may be missing) - ROBUST HANDLING
        if 'agent_state_change' in self.df.columns:
            # Combined QSE-agent data - extract behavioral fields (OLD NESTED FORMAT)
            try:
                time_series['goal_changes'] = self.df['agent_state_change'].apply(
                    lambda x: x['goal_changed'] if isinstance(x, dict) else False
                ).astype(int).values
                
                time_series['context_changes'] = self.df['agent_state_change'].apply(
                    lambda x: x['context_changed'] if isinstance(x, dict) else False
                ).astype(int).values
                
                time_series['q_value_changes'] = self.df['agent_state_change'].apply(
                    lambda x: x['q_value_change'] if isinstance(x, dict) else 0.0
                ).values
                
            except Exception as e:
                print(f"⚠️ Error extracting agent_state_change: {e}")
                # Fallback to zeros
                time_series['goal_changes'] = np.zeros(len(self.df), dtype=int)
                time_series['context_changes'] = np.zeros(len(self.df), dtype=int) 
                time_series['q_value_changes'] = np.zeros(len(self.df))
                
        elif 'goal' in self.df.columns and 'context' in self.df.columns:
            # NEW FLAT FORMAT - extract directly
            print("✅ Found flat format agent data")
            time_series['goal_changes'] = self.df['goal_changed'].astype(int).values if 'goal_changed' in self.df.columns else np.zeros(len(self.df), dtype=int)
            time_series['context_changes'] = self.df['context_changed'].astype(int).values if 'context_changed' in self.df.columns else np.zeros(len(self.df), dtype=int)
            time_series['q_value_changes'] = self.df['q_value_change'].values if 'q_value_change' in self.df.columns else np.zeros(len(self.df))
            
        else:
            # QSE-only data - generate behavioral proxies from QSE dynamics
            print("⚠️ No agent data found - generating behavioral proxies from QSE dynamics")
            
            # Context changes from phasic ruptures
            phasic_events = np.array(time_series['phasic_rupture'])
            time_series['context_changes'] = phasic_events.copy()

            # Goal changes from significant tau shifts  
            tau_diff = np.abs(np.diff(np.array(time_series['tau'])))
            tau_diff = np.append(tau_diff, 0)  # Same length
            time_series['goal_changes'] = (tau_diff > 0.1).astype(int)
            
            # Q-value changes from sigma dynamics
            time_series['q_value_changes'] = np.abs(np.array(time_series['sigma_mean'])) * 0.5
        
        # Decision events and QSE influence
        time_series['decision_events'] = self.df['decision_triggered'].astype(int).values if 'decision_triggered' in self.df.columns else np.array(time_series['goal_changes'])
        time_series['qse_influence'] = self.df['qse_influence_score'].values if 'qse_influence_score' in self.df.columns else np.abs(np.array(time_series['sigma_mean']))
        
        # Context evolution
        if 'agent_post_state' in self.df.columns:
            try:
                time_series['context_id'] = self.df['agent_post_state'].apply(
                    lambda x: x['context'] if isinstance(x, dict) else 0
                ).values
            except:
                time_series['context_id'] = np.cumsum(np.array(time_series['phasic_rupture'])) % 10
        elif 'context' in self.df.columns:
            # Flat format context
            time_series['context_id'] = self.df['context'].values
        else:
            # Generate context evolution from rupture events
            time_series['context_id'] = np.cumsum(np.array(time_series['phasic_rupture'])) % 10
        
        time_series = _ensure_behavioral_fields(time_series)
        return time_series
        
    def detect_regime_transitions(self, window_size: int = 50) -> List[Dict]:
        """Detect QSE regime transitions and their behavioral consequences"""
        
        if self.df is None:
            return []

        ts = self.extract_time_series()
        transitions = []
        
        # Detect tau regime changes (significant shifts in emergent time)
        tau_smooth = pd.Series(ts['tau']).rolling(window=window_size, center=True).mean()
        tau_changes = np.abs(np.diff(tau_smooth.dropna())) > 0.05  # Threshold for significant change
        
        # Detect sigma polarity flips
        sigma_sign_changes = np.diff(np.sign(ts['sigma_mean'])) != 0
        
        # Detect entropy regime shifts
        entropy_smooth = pd.Series(ts['entropy']).rolling(window=window_size, center=True).mean()
        entropy_changes = np.abs(np.diff(entropy_smooth.dropna())) > 0.1
        
        # Combine and analyze
        for i, (tau_change, sigma_flip) in enumerate(zip(tau_changes, sigma_sign_changes)):
            if tau_change or sigma_flip:
                # Look at behavioral consequences in next 20 steps
                start_idx = i + 1
                end_idx = min(start_idx + 20, len(ts['goal_changes']))
                
                post_behavior = {
                    'goal_changes': np.sum(ts['goal_changes'][start_idx:end_idx]),
                    'context_changes': np.sum(ts['context_changes'][start_idx:end_idx]),
                    'decision_events': np.sum(ts['decision_events'][start_idx:end_idx]),
                    'avg_qse_influence': np.mean(ts['qse_influence'][start_idx:end_idx])
                }
                
                transitions.append({
                    'step': start_idx,
                    'type': 'tau_regime' if tau_change else 'sigma_flip',
                    'tau_before': ts['tau'][i],
                    'tau_after': ts['tau'][i+1] if i+1 < len(ts['tau']) else ts['tau'][i],
                    'behavioral_response': post_behavior
                })
        
        return transitions
    
    def find_decision_chains(self, max_gap: int = 5) -> List[Dict]:
        """Find chains of decision events (decisions that trigger more decisions)"""
        
        ts = self.extract_time_series()
        decision_steps = np.where(ts['decision_events'] == 1)[0]
        
        chains = []
        current_chain = []
        
        for i, step in enumerate(decision_steps):
            if not current_chain:
                current_chain = [step]
            else:
                gap = step - current_chain[-1]
                if gap <= max_gap:
                    current_chain.append(step)
                else:
                    # End current chain, start new one
                    if len(current_chain) >= 3:  # Chains of 3+ decisions
                        chains.append({
                            'start_step': current_chain[0],
                            'end_step': current_chain[-1],
                            'length': len(current_chain),
                            'duration': current_chain[-1] - current_chain[0],
                            'steps': current_chain.copy()
                        })
                    current_chain = [step]
        
        # Don't forget the last chain
        if len(current_chain) >= 3:
            chains.append({
                'start_step': current_chain[0],
                'end_step': current_chain[-1],
                'length': len(current_chain),
                'duration': current_chain[-1] - current_chain[0],
                'steps': current_chain.copy()
            })
        
        return chains
    
    def analyze_context_effectiveness(self) -> Dict[str, Any]:
        """Analyze which contexts are most effective for different behaviors"""
        
        if self.df is None:
            return {}
        
        # Check if required columns exist
        if 'agent_post_state' not in self.df.columns or 'agent_state_change' not in self.df.columns:
            return {}
        
        context_analysis = defaultdict(lambda: defaultdict(list))
        
        for _, row in self.df.iterrows():
            context_id = row['agent_post_state']['context']
            state_change = row['agent_state_change']
            
            # Track outcomes by context
            context_analysis[context_id]['goal_changes'].append(state_change['goal_changed'])
            context_analysis[context_id]['q_value_changes'].append(state_change['q_value_change'])
            context_analysis[context_id]['qse_influence'].append(row['qse_influence_score'])
        
        # Summarize context effectiveness
        context_summary = {}
        for ctx_id, metrics in context_analysis.items():
            context_summary[ctx_id] = {
                'total_steps': len(metrics['goal_changes']),
                'goal_change_rate': np.mean(metrics['goal_changes']),
                'avg_q_value_change': np.mean(metrics['q_value_changes']),
                'avg_qse_influence': np.mean(metrics['qse_influence']),
                'effectiveness_score': np.mean(metrics['goal_changes']) * np.mean(metrics['qse_influence'])
            }
        
        return context_summary
    
    def detect_rare_events(self, percentile: float = 95) -> Dict[str, List]:
        """Detect rare/extreme events and their triggers"""
        
        if self.df is None:
            return {}
        
        ts = self.extract_time_series()
        rare_events = {}
        
        # High QSE influence events
        influence_threshold = np.percentile(ts['qse_influence'], percentile)
        high_influence_steps = np.where(ts['qse_influence'] > influence_threshold)[0]
        
        rare_events['high_qse_influence'] = []
        for step in high_influence_steps:
            if step > 0:  # Can look at previous step
                rare_events['high_qse_influence'].append({
                    'step': step,
                    'influence_score': ts['qse_influence'][step],
                    'tau_current': ts['tau'][step],
                    'phasic_rupture': bool(ts['phasic_rupture'][step]),
                    'context_id': ts['context_id'][step],
                    'triggered_decision': bool(ts['decision_events'][step])
                })
        
        # Rapid context switching (multiple context changes in short window)
        context_changes = ts['context_changes']
        rapid_switching = []
        window = 10
        for i in range(len(context_changes) - window):
            changes_in_window = np.sum(context_changes[i:i+window])
            if changes_in_window >= 3:  # 3+ context changes in 10 steps
                rapid_switching.append({
                    'start_step': i,
                    'context_changes': changes_in_window,
                    'avg_tau': np.mean(ts['tau'][i:i+window]),
                    'decisions_triggered': np.sum(ts['decision_events'][i:i+window])
                })
        
        rare_events['rapid_context_switching'] = rapid_switching
        
        return rare_events
    
    def compute_advanced_correlations(self) -> Dict[str, Any]:
        """Compute advanced correlation analysis"""
        
        if self.df is None:
            return {}
        
        ts = self.extract_time_series()
        correlations = {}
        
        # Multi-lag cross-correlations
        max_lag = 20
        qse_vars = ['tau', 'sigma_mean', 'phasic_rupture', 'entropy']
        behavior_vars = ['goal_changes', 'context_changes', 'q_value_changes']
        
        cross_corrs = {}
        for qse_var in qse_vars:
            for behavior_var in behavior_vars:
                corrs = []
                for lag in range(max_lag + 1):
                    if lag == 0:
                        corr = self._safe_corrcoef(ts[qse_var], ts[behavior_var])
                    else:
                        if len(ts[qse_var]) > lag:
                            corr = self._safe_corrcoef(ts[qse_var][:-lag], ts[behavior_var][lag:])
                        else:
                            corr = 0.0
                    corrs.append(corr)
                
                cross_corrs[f"{qse_var}_leads_{behavior_var}"] = {
                    'correlations': corrs,
                    'max_corr': np.nanmax(np.abs(corrs)),
                    'optimal_lag': np.nanargmax(np.abs(corrs))
                }
        
        correlations['cross_correlations'] = cross_corrs
        
        # Conditional correlations (QSE influence during different states)
        high_phasic = ts['phasic_rupture'] == 1
        low_phasic = ts['phasic_rupture'] == 0
        
        correlations['conditional'] = {
            'tau_vs_decisions_during_phasic': (
                self._safe_corrcoef(ts['tau'][high_phasic], ts['decision_events'][high_phasic]) 
                if np.sum(high_phasic) > 10 else 0.0
            ),
            'tau_vs_decisions_during_stable': (
                self._safe_corrcoef(ts['tau'][low_phasic], ts['decision_events'][low_phasic])
                if np.sum(low_phasic) > 10 else 0.0
            )
        }
                
        return correlations
    
    def run_complete_analysis(self, deep: bool = True) -> Dict[str, Any]:
        """Run complete analysis pipeline"""
        
        print("🔍 DEEP QSE DYNAMICS ANALYSIS")
        print("=" * 50)
        
        if not self.data:
            self.load_data()
        
        # Basic statistics
        print("📊 Computing basic statistics...")
        ts = self.extract_time_series()
        
        self.findings['basic_stats'] = {
            'total_steps': len(self.data),
            'decision_events': int(np.sum(ts['decision_events'])),
            'context_switches': int(np.sum(ts['context_changes'])),
            'goal_changes': int(np.sum(ts['goal_changes'])),
            'phasic_rupture_events': int(np.sum(ts['phasic_rupture'])),
            'avg_qse_influence': float(np.mean(ts['qse_influence'])),
            'max_qse_influence': float(np.max(ts['qse_influence']))
        }
        
        if deep:
            # Regime transitions
            print("🔄 Detecting regime transitions...")
            transitions = self.detect_regime_transitions()
            self.findings['regime_transitions'] = transitions
            
            # Decision chains
            print("🔗 Finding decision chains...")
            chains = self.find_decision_chains()
            self.findings['decision_chains'] = chains
            
            # Context effectiveness
            print("🎯 Analyzing context effectiveness...")
            context_eff = self.analyze_context_effectiveness()
            self.findings['context_effectiveness'] = context_eff
            
            # Rare events
            print("⚡ Detecting rare events...")
            rare_events = self.detect_rare_events()
            self.findings['rare_events'] = rare_events
            
            # Advanced correlations
            print("📈 Computing advanced correlations...")
            adv_corr = self.compute_advanced_correlations()
            self.findings['advanced_correlations'] = adv_corr
        
        # Generate insights
        self.findings['insights'] = self._generate_insights()
        
        return self.findings
    
    def _generate_insights(self) -> List[str]:
        """Generate key insights from analysis"""
        
        insights = []
        
        # Basic patterns
        stats = self.findings['basic_stats']
        decision_rate = stats['decision_events'] / stats['total_steps']
        context_rate = stats['context_switches'] / stats['total_steps']
        
        insights.append(f"🧠 Decision frequency: {decision_rate:.3f} decisions/step")
        insights.append(f"🔄 Context switching rate: {context_rate:.3f} switches/step")
        insights.append(f"⚡ Peak QSE influence: {stats['max_qse_influence']:.3f}")
        
        # Regime transitions
        if 'regime_transitions' in self.findings:
            transitions = self.findings['regime_transitions']
            if transitions:
                avg_response = np.mean([t['behavioral_response']['decision_events'] for t in transitions])
                insights.append(f"🔄 Regime transitions trigger {avg_response:.1f} decisions on average")
        
        # Decision chains
        if 'decision_chains' in self.findings:
            chains = self.findings['decision_chains']
            if chains:
                avg_length = np.mean([c['length'] for c in chains])
                insights.append(f"🔗 Found {len(chains)} decision chains, avg length {avg_length:.1f}")
        
        # Context effectiveness
        if 'context_effectiveness' in self.findings:
            ctx_eff = self.findings['context_effectiveness']
            best_contexts = sorted(ctx_eff.items(), key=lambda x: x[1]['effectiveness_score'], reverse=True)[:3]
            if best_contexts:
                best_ctx_id = best_contexts[0][0]
                best_score = best_contexts[0][1]['effectiveness_score']
                insights.append(f"🎯 Most effective context: {best_ctx_id} (score: {best_score:.3f})")
        
        # Advanced correlations
        if 'advanced_correlations' in self.findings:
            cross_corrs = self.findings['advanced_correlations']['cross_correlations']
            strongest = max(cross_corrs.items(), key=lambda x: x[1]['max_corr'])
            insights.append(f"🔗 Strongest lagged correlation: {strongest[0]} (r={strongest[1]['max_corr']:.3f}, lag={strongest[1]['optimal_lag']})")
        
        return insights
    
    def create_visualizations(self, save_dir: str = "qse_analysis_plots"):
        """Create comprehensive visualizations"""
        
        Path(save_dir).mkdir(exist_ok=True)
        ts = self.extract_time_series()
        
        # 1. Complete dynamics overview
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        steps = range(len(ts['tau']))
        
        # QSE dynamics
        ax1 = axes[0]
        ax1.plot(steps, ts['tau'], 'b-', alpha=0.7, label='τ (emergent time)')
        ax1.plot(steps, ts['sigma_mean'], 'r-', alpha=0.7, label='σ (curvature)')
        ax1.set_title('QSE Core Dynamics')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Rupture events
        ax2 = axes[1]
        rupture_steps = np.where(ts['phasic_rupture'] == 1)[0]
        ax2.scatter(rupture_steps, np.ones(len(rupture_steps)), c='red', s=20, alpha=0.7, label='Phasic Rupture')
        ax2.set_ylim(0, 2)
        ax2.set_title('Regime Transitions (Phasic Rupture Events)')
        ax2.legend()
        
        # Decision events and QSE influence
        ax3 = axes[2]
        decision_steps = np.where(ts['decision_events'] == 1)[0]
        ax3.scatter(decision_steps, np.ones(len(decision_steps)), c='green', s=15, alpha=0.6, label='Decisions')
        ax3_twin = ax3.twinx()
        ax3_twin.plot(steps, ts['qse_influence'], 'purple', alpha=0.7, label='QSE Influence')
        ax3.set_ylim(0, 2)
        ax3.set_title('Decision Events & QSE Influence')
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        
        # Context evolution
        ax4 = axes[3]
        ax4.plot(steps, ts['context_id'], 'orange', linewidth=1, alpha=0.8)
        ax4.set_title('Context Evolution')
        ax4.set_xlabel('Time Steps')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/qse_dynamics_overview.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Decision chain visualization
        if 'decision_chains' in self.findings:
            chains = self.findings['decision_chains']
            if chains:
                plt.figure(figsize=(12, 6))
                for i, chain in enumerate(chains[:10]):  # Show first 10 chains
                    plt.scatter(chain['steps'], [i] * len(chain['steps']), 
                               s=30, alpha=0.7, label=f"Chain {i+1} (len={chain['length']})")
                
                plt.xlabel('Time Steps')
                plt.ylabel('Decision Chain ID')
                plt.title('Decision Chains Over Time')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                plt.savefig(f"{save_dir}/decision_chains.png", dpi=300, bbox_inches='tight')
                plt.show()
        
        print(f"📊 Visualizations saved to {save_dir}/")
    
    def export_findings(self, output_file: str = "qse_dynamics_analysis.json"):
        """Export findings to JSON"""
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        json_findings = convert_for_json(self.findings)
        
        with open(output_file, 'w') as f:
            json.dump(json_findings, f, indent=2)
        
        print(f"📁 Findings exported to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze QSE Dynamics JSONL data')
    parser.add_argument('files', nargs='+', help='JSONL files to analyze')
    parser.add_argument('--deep', action='store_true', help='Run deep analysis')
    parser.add_argument('--export-findings', action='store_true', help='Export findings to JSON')
    parser.add_argument('--no-plots', action='store_true', help='Skip visualization')
    parser.add_argument('--output', default='qse_analysis_results.json', help='Output file for findings')
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = QSEDynamicsAnalyzer(args.files)
    findings = analyzer.run_complete_analysis(deep=args.deep)
    
    # Print insights
    print("\n🔍 KEY INSIGHTS:")
    print("=" * 40)
    for insight in findings['insights']:
        print(f"  {insight}")
    
    # Create visualizations
    if not args.no_plots:
        print("\n📊 Creating visualizations...")
        analyzer.create_visualizations()
    
    # Export findings
    if args.export_findings:
        analyzer.export_findings(args.output)
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
