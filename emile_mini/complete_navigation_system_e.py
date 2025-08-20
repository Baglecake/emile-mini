
#!/usr/bin/env python3
"""
Real QSE Autopoietic Validation Using Complete Navigation System
==============================================================

FIXED to use complete_navigation_system_d.py instead of basic embodied_qse_emile

Uses your existing sophisticated QSE analysis tools:
- qse_core_metric_runner_c.py for deep QSE dynamics  
- complete_navigation_system_d.py for QSE-behavior coupling
- analyze_qse_dynamics.py for pattern discovery

Usage:
    python real_qse_validation_existing_tools.py --comprehensive
    python real_qse_validation_existing_tools.py --qse-only --steps 50000
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
import time
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# Import your existing QSE tools
try:
    from emile_mini.qse_core_metric_runner_c import run_qse_metrics_collection
    QSE_CORE_AVAILABLE = True
    print("✅ QSE Core Metrics Runner available")
except ImportError:
    QSE_CORE_AVAILABLE = False
    print("⚠️ qse_core_metric_runner_c.py not found")

try:
    from emile_mini.qse_agent_dynamics_runner import QSEAgentDynamicsRunner
    QSE_AGENT_AVAILABLE = True
    print("✅ QSE Agent Dynamics Runner available")
except ImportError:
    QSE_AGENT_AVAILABLE = False
    print("⚠️ qse_agent_dynamics_runner.py not found")

try:
    from emile_mini.analyze_qse_dynamics import QSEDynamicsAnalyzer
    QSE_ANALYZER_AVAILABLE = True
    print("✅ QSE Dynamics Analyzer available")
except ImportError:
    QSE_ANALYZER_AVAILABLE = False
    print("⚠️ analyze_qse_dynamics.py not found")

# FIXED: Use complete navigation system instead of basic embodied
try:
    from emile_mini.complete_navigation_system_d import ProactiveEmbodiedQSEAgent, ClearPathEnvironment, test_complete_navigation
    COMPLETE_NAV_AVAILABLE = True
    print("✅ Complete Navigation System available")
except ImportError:
    COMPLETE_NAV_AVAILABLE = False
    print("⚠️ complete_navigation_system_d.py not found")


class RealQSEAutopoieticValidator:
    """Validates autopoiesis using complete navigation system and existing QSE tools"""

    def __init__(self, output_dir: str = "real_qse_validation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.results = {}

        print(f"🧬 REAL QSE AUTOPOIETIC VALIDATION")
        print(f"📁 Output: {self.output_dir}")
        print(f"🔬 Using complete navigation system + existing QSE tools")

    def run_qse_core_analysis(self, steps: int = 50000, seed: int = 2024) -> Dict[str, Any]:
        """Run QSE core dynamics analysis using your sophisticated tool"""
        print(f"\n🔬 Running QSE Core Analysis ({steps} steps, seed={seed})")

        if not QSE_CORE_AVAILABLE:
            print("⚠️ QSE Core Metrics Runner not available")
            return {'available': False}

        # Generate QSE core metrics using your tool
        qse_output = self.output_dir / f"qse_core_metrics_{steps}.jsonl"
        summary_file = run_qse_metrics_collection(
            steps=steps,
            output_file=str(qse_output),
            verbose=True,
            seed=seed
        )

        # Load the summary
        if isinstance(summary_file, str) and Path(summary_file).exists():
            with open(summary_file, 'r') as f:
                qse_summary = json.load(f)
        else:
            qse_summary = {}

        # Extract autopoietic evidence
        autopoietic_evidence = self._extract_autopoietic_evidence_from_qse_core(qse_summary)

        return {
            'available': True,
            'qse_summary': qse_summary,
            'autopoietic_evidence': autopoietic_evidence,
            'data_file': str(qse_output),
            'summary_file': summary_file
        }

    def run_qse_agent_coupling_analysis(self, steps: int = 15000, agent_type: str = "cognitive") -> Dict[str, Any]:
        """Run QSE-agent coupling analysis using your tool"""
        print(f"\n🤖 Running QSE-Agent Coupling Analysis ({agent_type}, {steps} steps)")

        if not QSE_AGENT_AVAILABLE:
            print("⚠️ QSE Agent Dynamics Runner not available")
            return {'available': False}

        try:
            # FIXED: Use complete navigation system for embodied type
            if agent_type == "embodied" and COMPLETE_NAV_AVAILABLE:
                env_type = "complete_navigation"  # Use enhanced navigation environment
            else:
                env_type = "embodied" if agent_type == "embodied" else "basic"
                
            runner = QSEAgentDynamicsRunner(agent_type=agent_type, environment_type=env_type)

            output_file = self.output_dir / f"qse_agent_dynamics_{agent_type}.jsonl"

            results = runner.run_dynamics_analysis(
                steps=steps,
                output_file=str(output_file),
                seed=2024
            )

            # Extract coupling evidence
            coupling_evidence = self._extract_coupling_evidence(results)

            return {
                'available': True,
                'dynamics_results': results,
                'coupling_evidence': coupling_evidence,
                'data_file': str(output_file)
            }

        except Exception as e:
            print(f"⚠️ QSE-Agent coupling analysis failed: {e}")
            return {'available': False, 'error': str(e)}



    def run_advanced_pattern_analysis(self, jsonl_file: str) -> Dict[str, Any]:
        """Run advanced pattern analysis using your analyzer"""
        print(f"\n🔍 Running Advanced Pattern Analysis on {jsonl_file}")

        if not QSE_ANALYZER_AVAILABLE:
            print("⚠️ QSE Dynamics Analyzer not available")
            return {'available': False}

        if not Path(jsonl_file).exists():
            print(f"⚠️ Data file not found: {jsonl_file}")
            return {'available': False}

        try:
            # Use your sophisticated analyzer
            analyzer = QSEDynamicsAnalyzer([jsonl_file])
            analyzer.load_data()
            findings = analyzer.run_complete_analysis(deep=True)

            # Extract pattern evidence
            pattern_evidence = self._extract_pattern_evidence(findings)

            print(f"✅ Found {len(findings.get('decision_chains', []))} decision chains")
            print(f"✅ Found {len(findings.get('regime_transitions', []))} regime transitions")

            return {
                'available': True,
                'findings': findings,
                'pattern_evidence': pattern_evidence
            }

        except Exception as e:
            print(f"⚠️ Advanced pattern analysis failed: {e}")
            return {'available': False, 'error': str(e)}

    def _extract_coupling_evidence(self, dynamics_results: Dict) -> Dict[str, Any]:
        """Extract QSE-behavior coupling evidence - ROBUST VERSION"""

        findings = dynamics_results.get('key_findings', {})

        # Handle case where findings might be empty or missing
        if not findings:
            print("⚠️ No findings in dynamics results - generating minimal evidence")
            coupling_evidence = {
                'autopoietic_health': {'overall_assessment': 'unknown'},
                'agent_responsiveness': {'coupling_strength': 'insufficient_data'},
                'causal_insights': {'total_relationships': 0, 'interpretation': 'No data available'},
                'coupling_strength': 'WEAK QSE-behavior coupling with no data available'
            }
        else:
            coupling_evidence = {
                'autopoietic_health': findings.get('autopoietic_health', {'overall_assessment': 'unknown'}),
                'agent_responsiveness': findings.get('agent_responsiveness', {'coupling_strength': 'unknown'}),
                'causal_insights': findings.get('causal_insights', {'total_relationships': 0}),
                'coupling_strength': self._assess_coupling_strength(findings)
            }

        return coupling_evidence

    def _assess_coupling_strength(self, findings: Dict) -> str:
        """Assess QSE-behavior coupling strength"""

        health = findings.get('autopoietic_health', {})
        responsiveness = findings.get('agent_responsiveness', {})
        causal = findings.get('causal_insights', {})

        overall_health = health.get('overall_assessment', 'unknown')
        coupling_strength = responsiveness.get('coupling_strength', 'unknown')
        causal_relationships = causal.get('total_relationships', 0)

        if overall_health == 'healthy' and coupling_strength == 'strong' and causal_relationships > 2:
            return "STRONG QSE-behavior coupling with healthy autopoietic dynamics"
        elif coupling_strength in ['strong', 'moderate'] and causal_relationships > 0:
            return "MODERATE QSE-behavior coupling with some causal relationships"
        else:
            return "WEAK QSE-behavior coupling with limited causal evidence"

    def run_comprehensive_validation(self, qse_steps: int = 50000, agent_steps: int = 15000) -> Dict[str, Any]:
        """Run comprehensive validation using all your existing tools"""

        print(f"🧬 COMPREHENSIVE REAL QSE AUTOPOIETIC VALIDATION")
        print(f"=" * 70)
        print(f"Using your existing sophisticated QSE analysis tools")

        start_time = time.time()
        results = {}

        # 1. QSE Core Analysis
        print(f"\n1/4 QSE Core Dynamics Analysis...")
        qse_core_results = self.run_qse_core_analysis(steps=qse_steps)
        results['qse_core'] = qse_core_results

        # 2. QSE-Agent Coupling (Cognitive)
        print(f"\n2/4 QSE-Agent Coupling (Cognitive)...")
        cognitive_results = self.run_qse_agent_coupling_analysis(steps=agent_steps, agent_type="cognitive")
        results['cognitive_coupling'] = cognitive_results

        # 3. QSE-Agent Coupling (Embodied) - FIXED to use complete navigation
        print(f"\n3/4 QSE-Agent Coupling (Embodied with Complete Navigation)...")
        embodied_results = self.run_qse_agent_coupling_analysis(steps=agent_steps//2, agent_type="embodied")
        results['embodied_coupling'] = embodied_results

        # 4. Advanced Pattern Analysis
        print(f"\n4/4 Advanced Pattern Analysis...")
        pattern_results = {}

        # Analyze QSE core data if available
        if qse_core_results.get('available') and 'data_file' in qse_core_results:
            pattern_results['qse_core'] = self.run_advanced_pattern_analysis(qse_core_results['data_file'])

        # Analyze cognitive coupling data if available
        if cognitive_results.get('available') and 'data_file' in cognitive_results:
            pattern_results['cognitive'] = self.run_advanced_pattern_analysis(cognitive_results['data_file'])

        # Analyze embodied coupling data if available
        if embodied_results.get('available') and 'data_file' in embodied_results:
            pattern_results['embodied'] = self.run_advanced_pattern_analysis(embodied_results['data_file'])

        results['pattern_analysis'] = pattern_results

        # Comprehensive Assessment
        overall_assessment = self._generate_comprehensive_assessment(results)
        publication_ready = self._assess_publication_readiness(results)

        final_results = {
            'validation_timestamp': time.time(),
            'total_runtime_minutes': (time.time() - start_time) / 60,
            'methodology': 'real_qse_existing_tools_with_complete_navigation',
            'analysis_results': results,
            'overall_assessment': overall_assessment,
            'publication_ready': publication_ready,
            'autopoiesis_score': self._calculate_comprehensive_autopoiesis_score(results)
        }

        # Save results
        self._save_results(final_results)

        # Print summary
        self._print_comprehensive_summary(final_results)

        return final_results

    # Include all the existing helper methods from original file
    def _extract_autopoietic_evidence_from_qse_core(self, qse_summary: Dict) -> Dict[str, Any]:
        """Extract autopoietic evidence from QSE core summary"""

        if not qse_summary:
            return {'evidence_quality': 'no_data'}

        # Extract autopoiesis markers
        markers = qse_summary.get('autopoiesis_markers', {})
        gates = qse_summary.get('autopoiesis_gates', {})
        rupture_analysis = qse_summary.get('rupture_episode_analysis', {})

        evidence = {
            'self_organization': {
                'index': markers.get('self_organization_index', 0),
                'strength': min(1.0, markers.get('self_organization_index', 0) * 10),
                'assessment': self._assess_self_organization(markers.get('self_organization_index', 0))
            },

            'viability_maintenance': {
                'score': markers.get('viability_maintenance_score', 0),
                'strength': markers.get('viability_maintenance_score', 0),
                'assessment': self._assess_viability(markers.get('viability_maintenance_score', 0))
            },

            'productive_instability': {
                'balance': markers.get('productive_instability_balance', 0),
                'tonic_aliveness': markers.get('tonic_aliveness_score', 0),
                'phasic_dynamics': rupture_analysis.get('phasic_rupture_time_fraction', 0),
                'strength': self._calculate_instability_strength(markers, rupture_analysis),
                'assessment': self._assess_productive_instability(markers, rupture_analysis)
            },

            'boundary_integrity': {
                'score': markers.get('boundary_integrity_score', 0),
                'strength': markers.get('boundary_integrity_score', 0),
                'assessment': self._assess_boundary_integrity(markers.get('boundary_integrity_score', 0))
            },

            'temporal_autonomy': {
                'index': markers.get('temporal_autonomy_index', 0),
                'strength': markers.get('temporal_autonomy_index', 0),
                'assessment': self._assess_temporal_autonomy(markers.get('temporal_autonomy_index', 0))
            },

            'autopoiesis_gates': {
                'total_gates': len(gates),
                'gates_passed': sum(1 for passed in gates.values() if passed),
                'gates_failed': sum(1 for passed in gates.values() if not passed),
                'gate_success_rate': sum(1 for passed in gates.values() if passed) / len(gates) if gates else 0,
                'all_gates_pass': all(gates.values()) if gates else False,
                'assessment': self._assess_gates(gates)
            }
        }

        # Overall autopoietic assessment
        evidence['overall_assessment'] = self._compute_overall_autopoietic_assessment(evidence)

        return evidence

    def _extract_pattern_evidence(self, findings: Dict) -> Dict[str, Any]:
        """Extract autopoietic pattern evidence from analyzer findings"""

        pattern_evidence = {
            'decision_chains': {
                'total_chains': len(findings.get('decision_chains', [])),
                'avg_length': np.mean([c['length'] for c in findings.get('decision_chains', [])]) if findings.get('decision_chains') else 0,
                'max_length': max([c['length'] for c in findings.get('decision_chains', [])]) if findings.get('decision_chains') else 0,
                'recursive_evidence': len(findings.get('decision_chains', [])) > 0
            },

            'regime_transitions': {
                'total_transitions': len(findings.get('regime_transitions', [])),
                'behavioral_responses': [t.get('behavioral_response', {}) for t in findings.get('regime_transitions', [])],
                'autonomy_evidence': len(findings.get('regime_transitions', [])) > 0
            },

            'context_effectiveness': findings.get('context_effectiveness', {}),

            'rare_events': findings.get('rare_events', {}),

            'advanced_correlations': findings.get('advanced_correlations', {})
        }

        return pattern_evidence

    # Include all the assessment helper methods from the original
    def _assess_self_organization(self, index: float) -> str:
        if index > 0.15:
            return "STRONG self-organization with high temporal variability"
        elif index > 0.08:
            return "MODERATE self-organization with structured dynamics"
        elif index > 0.03:
            return "WEAK self-organization with some temporal structure"
        else:
            return "NO significant self-organization detected"

    def _assess_viability(self, score: float) -> str:
        if score > 0.99:
            return "EXCELLENT viability maintenance with minimal boundary violations"
        elif score > 0.95:
            return "GOOD viability maintenance with occasional violations"
        elif score > 0.85:
            return "MODERATE viability with some boundary stress"
        else:
            return "POOR viability with frequent boundary violations"

    def _calculate_instability_strength(self, markers: Dict, rupture_analysis: Dict) -> float:
        phasic_balance = markers.get('productive_instability_balance', 0)
        tonic_aliveness = markers.get('tonic_aliveness_score', 0)

        if tonic_aliveness > 0.9 and 0.1 <= phasic_balance <= 0.4:
            return min(1.0, tonic_aliveness + phasic_balance)
        else:
            return max(0.0, min(1.0, (tonic_aliveness + phasic_balance) / 2))

    def _assess_productive_instability(self, markers: Dict, rupture_analysis: Dict) -> str:
        tonic_score = markers.get('tonic_aliveness_score', 0)
        phasic_balance = markers.get('productive_instability_balance', 0)

        if tonic_score > 0.95 and 0.15 <= phasic_balance <= 0.35:
            return "EXCELLENT productive instability with healthy tonic aliveness and phasic dynamics"
        elif tonic_score > 0.8 and 0.05 <= phasic_balance <= 0.5:
            return "GOOD productive instability with adequate background activity"
        elif tonic_score > 0.5:
            return "MODERATE productive instability with some background activity"
        else:
            return "POOR productive instability - insufficient background aliveness"

    def _assess_boundary_integrity(self, score: float) -> str:
        if score > 0.9999:
            return "EXCEPTIONAL boundary integrity with virtually no violations"
        elif score > 0.999:
            return "EXCELLENT boundary integrity with minimal violations"
        elif score > 0.99:
            return "GOOD boundary integrity with occasional violations"
        else:
            return "POOR boundary integrity with frequent violations"

    def _assess_temporal_autonomy(self, index: float) -> str:
        if index > 0.99:
            return "STRONG temporal autonomy - rarely saturated at maximum"
        elif index > 0.95:
            return "GOOD temporal autonomy with occasional saturation"
        elif index > 0.8:
            return "MODERATE temporal autonomy with some saturation"
        else:
            return "WEAK temporal autonomy - frequently saturated"

    def _assess_gates(self, gates: Dict) -> str:
        if not gates:
            return "NO gate data available"

        passed = sum(1 for passed in gates.values() if passed)
        total = len(gates)

        if passed == total:
            return f"ALL {total} autopoiesis gates PASS - system meets all criteria"
        elif passed >= total * 0.8:
            return f"{passed}/{total} gates pass - mostly healthy system"
        else:
            return f"ONLY {passed}/{total} gates pass - system needs attention"

    def _compute_overall_autopoietic_assessment(self, evidence: Dict) -> Dict[str, Any]:
        # Extract component scores
        self_org_strength = evidence['self_organization']['strength']
        viability_strength = evidence['viability_maintenance']['strength']
        instability_strength = evidence['productive_instability']['strength']
        boundary_strength = evidence['boundary_integrity']['strength']
        temporal_strength = evidence['temporal_autonomy']['strength']
        gates_score = evidence['autopoiesis_gates']['gate_success_rate']

        # Weighted combination
        weights = [0.15, 0.2, 0.25, 0.15, 0.1, 0.15]
        scores = [self_org_strength, viability_strength, instability_strength, boundary_strength, temporal_strength, gates_score]

        overall_score = sum(w * s for w, s in zip(weights, scores))

        return {
            'component_scores': {
                'self_organization': self_org_strength,
                'viability_maintenance': viability_strength,
                'productive_instability': instability_strength,
                'boundary_integrity': boundary_strength,
                'temporal_autonomy': temporal_strength,
                'autopoiesis_gates': gates_score
            },
            'overall_autopoiesis_score': overall_score,
            'autopoietic_classification': (
                'EXCEPTIONAL' if overall_score > 0.9 else
                'STRONG' if overall_score > 0.75 else
                'GOOD' if overall_score > 0.6 else
                'MODERATE' if overall_score > 0.4 else
                'WEAK'
            ),
            'evidence_quality': 'authentic_qse_complete_navigation_dynamics'
        }

    def _generate_comprehensive_assessment(self, results: Dict) -> str:
        """Generate comprehensive assessment"""

        assessments = []

        # QSE Core assessment
        if results.get('qse_core', {}).get('available'):
            qse_evidence = results['qse_core']['autopoietic_evidence']
            overall = qse_evidence.get('overall_assessment', {})
            classification = overall.get('autopoietic_classification', 'UNKNOWN')
            assessments.append(f"QSE Core: {classification}")

        # Coupling assessments
        if results.get('cognitive_coupling', {}).get('available'):
            coupling = results['cognitive_coupling']['coupling_evidence']
            strength = coupling.get('coupling_strength', 'UNKNOWN')
            assessments.append(f"Cognitive coupling: {strength}")

        if results.get('embodied_coupling', {}).get('available'):
            coupling = results['embodied_coupling']['coupling_evidence']
            strength = coupling.get('coupling_strength', 'UNKNOWN')
            assessments.append(f"Embodied coupling (Complete Navigation): {strength}")

        # Pattern analysis
        patterns = results.get('pattern_analysis', {})
        if patterns:
            pattern_assessments = []
            for analysis_type, data in patterns.items():
                if data.get('available'):
                    chains = data.get('pattern_evidence', {}).get('decision_chains', {})
                    if chains.get('total_chains', 0) > 0:
                        pattern_assessments.append(f"{chains['total_chains']} decision chains")

            if pattern_assessments:
                assessments.append(f"Patterns: {', '.join(pattern_assessments)}")

        if assessments:
            return f"COMPREHENSIVE: {' | '.join(assessments)}"
        else:
            return "INSUFFICIENT DATA for comprehensive assessment"

    def _assess_publication_readiness(self, results: Dict) -> bool:
        """Assess publication readiness"""

        criteria_met = 0

        # QSE core evidence
        if results.get('qse_core', {}).get('available'):
            qse_evidence = results['qse_core']['autopoietic_evidence']
            overall = qse_evidence.get('overall_assessment', {})
            if overall.get('overall_autopoiesis_score', 0) > 0.75:
                criteria_met += 2  # Strong QSE core evidence worth 2 points

            gates = qse_evidence.get('autopoiesis_gates', {})
            if gates.get('all_gates_pass', False):
                criteria_met += 1  # All gates passing worth 1 point

        # Coupling evidence
        for coupling_type in ['cognitive_coupling', 'embodied_coupling']:
            if results.get(coupling_type, {}).get('available'):
                coupling = results[coupling_type]['coupling_evidence']
                strength = coupling.get('coupling_strength', '')
                if 'STRONG' in strength or 'MODERATE' in strength:
                    criteria_met += 1

        # Pattern evidence
        patterns = results.get('pattern_analysis', {})
        pattern_evidence_count = 0
        for analysis_type, data in patterns.items():
            if data.get('available'):
                evidence = data.get('pattern_evidence', {})
                if evidence.get('decision_chains', {}).get('total_chains', 0) > 0:
                    pattern_evidence_count += 1
                if evidence.get('regime_transitions', {}).get('total_transitions', 0) > 0:
                    pattern_evidence_count += 1

        if pattern_evidence_count >= 2:
            criteria_met += 1

        # Need at least 4 out of 6 possible criteria
        return criteria_met >= 4

    def _calculate_comprehensive_autopoiesis_score(self, results: Dict) -> float:
        """Calculate comprehensive autopoiesis score"""

        scores = []
        weights = []

        # QSE Core (most important)
        if results.get('qse_core', {}).get('available'):
            qse_evidence = results['qse_core']['autopoietic_evidence']
            overall = qse_evidence.get('overall_assessment', {})
            score = overall.get('overall_autopoiesis_score', 0)
            scores.append(score)
            weights.append(0.5)  # 50% weight for core QSE

        # Coupling evidence
        coupling_scores = []
        for coupling_type in ['cognitive_coupling', 'embodied_coupling']:
            if results.get(coupling_type, {}).get('available'):
                # Simple coupling score based on health and responsiveness
                health = results[coupling_type]['coupling_evidence'].get('autopoietic_health', {})
                responsiveness = results[coupling_type]['coupling_evidence'].get('agent_responsiveness', {})

                health_score = 1.0 if health.get('overall_assessment') == 'healthy' else 0.5
                resp_score = {'strong': 1.0, 'moderate': 0.7, 'weak': 0.3}.get(responsiveness.get('coupling_strength', 'weak'), 0.3)

                coupling_scores.append((health_score + resp_score) / 2)

        if coupling_scores:
            scores.append(np.mean(coupling_scores))
            weights.append(0.3)  # 30% weight for coupling

        # Pattern evidence
        patterns = results.get('pattern_analysis', {})
        pattern_scores = []
        for analysis_type, data in patterns.items():
            if data.get('available'):
                evidence = data.get('pattern_evidence', {})
                chains = evidence.get('decision_chains', {})
                transitions = evidence.get('regime_transitions', {})

                chain_score = min(1.0, chains.get('total_chains', 0) / 5.0)
                transition_score = min(1.0, transitions.get('total_transitions', 0) / 10.0)

                pattern_scores.append((chain_score + transition_score) / 2)

        if pattern_scores:
            scores.append(np.mean(pattern_scores))
            weights.append(0.2)  # 20% weight for patterns

        # Weighted average
        if scores:
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize
            return float(np.average(scores, weights=weights))
        else:
            return 0.0

    def _save_results(self, results: Dict):
        """Save results to JSON"""
        output_file = self.output_dir / "real_qse_validation_results.json"

        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, dict):
                return {str(k): convert_for_json(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [convert_for_json(x) for x in obj]
            return str(obj)

        json_results = convert_for_json(results)

        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)

        print(f"\n📁 Results saved to: {output_file}")

    def _print_comprehensive_summary(self, results: Dict):
        """Print comprehensive summary"""

        print(f"\n" + "=" * 70)
        print(f"🧬 REAL QSE AUTOPOIETIC VALIDATION SUMMARY")
        print(f"=" * 70)

        score = results['autopoiesis_score']
        assessment = results['overall_assessment']
        pub_ready = results['publication_ready']
        runtime = results['total_runtime_minutes']

        print(f"🔬 Overall Autopoiesis Score: {score:.3f}")
        print(f"📋 Assessment: {assessment}")
        print(f"⏱️ Runtime: {runtime:.1f} minutes")
        print(f"🔬 Methodology: {results['methodology']}")

        print(f"\n📊 DETAILED ANALYSIS RESULTS:")

        # QSE Core Results
        if results['analysis_results'].get('qse_core', {}).get('available'):
            qse_core = results['analysis_results']['qse_core']
            evidence = qse_core['autopoietic_evidence']
            overall = evidence.get('overall_assessment', {})

            print(f"\n🔬 QSE Core Dynamics:")
            print(f"  Score: {overall.get('overall_autopoiesis_score', 0):.3f}")
            print(f"  Classification: {overall.get('autopoietic_classification', 'UNKNOWN')}")

            # Component assessments
            for component in ['self_organization', 'viability_maintenance', 'productive_instability',
                            'boundary_integrity', 'temporal_autonomy', 'autopoiesis_gates']:
                comp_data = evidence.get(component, {})
                assessment = comp_data.get('assessment', 'No assessment')
                print(f"    {component.replace('_', ' ').title()}: {assessment}")

        # Coupling Results
        for coupling_type in ['cognitive_coupling', 'embodied_coupling']:
            if results['analysis_results'].get(coupling_type, {}).get('available'):
                coupling = results['analysis_results'][coupling_type]
                evidence = coupling['coupling_evidence']

                display_name = coupling_type.replace('_', ' ').title()
                if coupling_type == 'embodied_coupling':
                    display_name += " (Complete Navigation)"

                print(f"\n🤖 {display_name}:")
                print(f"  Coupling Strength: {evidence.get('coupling_strength', 'Unknown')}")

                if 'causal_insights' in evidence and evidence['causal_insights']:
                    causal = evidence['causal_insights']
                    print(f"  Strongest Relationship: {causal.get('interpretation', 'None found')}")
                    print(f"  Total Causal Relationships: {causal.get('total_relationships', 0)}")

        # Pattern Results
        patterns = results['analysis_results'].get('pattern_analysis', {})
        if patterns:
            print(f"\n🔍 Pattern Analysis:")
            for analysis_type, data in patterns.items():
                if data.get('available'):
                    evidence = data.get('pattern_evidence', {})
                    chains = evidence.get('decision_chains', {})
                    transitions = evidence.get('regime_transitions', {})

                    print(f"  {analysis_type.title()}:")
                    print(f"    Decision Chains: {chains.get('total_chains', 0)} (max length: {chains.get('max_length', 0)})")
                    print(f"    Regime Transitions: {transitions.get('total_transitions', 0)}")

        print(f"\n🎯 PUBLICATION READINESS:")
        if pub_ready:
            print(f"✅ PUBLICATION READY")
            print(f"✅ Strong evidence across multiple analysis dimensions")
            print(f"✅ Using authentic QSE dynamics from sophisticated tools")
            print(f"✅ Enhanced embodied coupling with complete navigation system")
            print(f"✅ Comprehensive multi-modal validation")
        else:
            print(f"⚠️ Additional evidence recommended")
            print(f"📈 Consider longer runs or additional analysis")

        print(f"\n📁 All results saved to: {self.output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Real QSE Autopoietic Validation Using Existing Tools + Complete Navigation')
    parser.add_argument('--comprehensive', action='store_true',
                       help='Run comprehensive validation using all tools')
    parser.add_argument('--qse-only', action='store_true',
                       help='Run only QSE core analysis')
    parser.add_argument('--qse-steps', type=int, default=50000,
                       help='Steps for QSE core analysis')
    parser.add_argument('--agent-steps', type=int, default=15000,
                       help='Steps for QSE-agent coupling analysis')
    parser.add_argument('--output', type=str, default='real_qse_validation',
                       help='Output directory')

    args = parser.parse_args()

    validator = RealQSEAutopoieticValidator(output_dir=args.output)

    if args.comprehensive:
        results = validator.run_comprehensive_validation(
            qse_steps=args.qse_steps,
            agent_steps=args.agent_steps
        )

    elif args.qse_only:
        print(f"🔬 Running QSE Core Analysis Only...")
        results = validator.run_qse_core_analysis(steps=args.qse_steps)

        if results.get('available'):
            evidence = results['autopoietic_evidence']
            overall = evidence.get('overall_assessment', {})

            print(f"\n🧬 QSE Core Autopoietic Evidence:")
            print(f"Score: {overall.get('overall_autopoiesis_score', 0):.3f}")
            print(f"Classification: {overall.get('autopoietic_classification', 'UNKNOWN')}")

            gates = evidence.get('autopoiesis_gates', {})
            print(f"Gates: {gates.get('gates_passed', 0)}/{gates.get('total_gates', 0)} passed")

    else:
        print("Please specify --comprehensive or --qse-only")
        print("Example: python real_qse_validation_existing_tools.py --comprehensive")


if __name__ == "__main__":
    main()
