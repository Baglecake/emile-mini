#!/usr/bin/env python3
"""
Test JUST the cognitive coupling analysis to verify flat format fix
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

# Import the validation class
from real_qse_validation_existing_tools import RealQSEAutopoieticValidator

def test_cognitive_coupling_only():
    """Test just the cognitive coupling analysis"""
    
    print("🧪 TESTING COGNITIVE COUPLING ONLY")
    print("=" * 50)
    
    # Create validation framework
    validator = RealQSEAutopoieticValidator()
    
    # Run JUST the cognitive coupling analysis
    print("🤖 Running cognitive coupling analysis...")
    cognitive_results = validator.run_qse_agent_coupling_analysis(
        steps=1000,  # Much smaller for quick test
        agent_type="cognitive", 
        seed=2024
    )
    
    print("\n📊 COGNITIVE COUPLING RESULTS:")
    print(f"Available: {cognitive_results.get('available', 'Unknown')}")
    print(f"Data file: {cognitive_results.get('data_file', 'None')}")
    
    if cognitive_results.get('available'):
        # Test if analyzer can parse the data
        print("\n🔍 Testing pattern analysis on cognitive data...")
        pattern_results = validator.run_advanced_pattern_analysis(
            cognitive_results['data_file']
        )
        
        print(f"Pattern analysis available: {pattern_results.get('available', False)}")
        if pattern_results.get('available'):
            findings = pattern_results.get('findings', {})
            print(f"Decision chains found: {len(findings.get('decision_chains', []))}")
            print(f"Regime transitions: {len(findings.get('regime_transitions', []))}")
            print("✅ SUCCESS: Analyzer found real agent data!")
        else:
            print(f"❌ FAILED: {pattern_results.get('error', 'Unknown error')}")
    else:
        print(f"❌ FAILED: {cognitive_results.get('error', 'Unknown error')}")

if __name__ == "__main__":
    test_cognitive_coupling_only()