#!/usr/bin/env python3
"""
Usage examples for the multimodal pathway feature.
These examples demonstrate how to use the new multimodal functionality.
"""

import numpy as np
from emile_mini.agent import EmileAgent
from emile_mini.config import CONFIG
from emile_mini.embodied_qse_emile import EmbodiedQSEAgent, EmbodiedEnvironment

def example_basic_multimodal():
    """Example 1: Basic multimodal agent with transient attention."""
    print("📖 Example 1: Basic multimodal agent with transient attention")
    
    # Enable multimodal functionality
    CONFIG.MULTIMODAL_ENABLED = True
    agent = EmileAgent(CONFIG)
    agent.goal.add_goal('explore')
    
    # Set attention mode to 'listening' for 50 steps
    agent.set_attention_mode('listening', steps=50)
    print(f"Set attention to 'listening' for 50 steps")
    
    for t in range(100):
        # Provide audio input occasionally
        external_input = {}
        if t % 10 == 0:
            external_input['audio'] = np.random.randn(16000) * 0.01
            print(f"Step {t}: Provided audio input, attention={agent.attention_mode}")
        
        result = agent.step(dt=0.01, external_input=external_input)
        
        # Print progress every 20 steps
        if t % 20 == 0:
            print(f"Step {t}: attention={agent.attention_mode}, steps_left={agent._attention_steps_left}, sigma={result['sigma_mean']:.3f}")
    
    print("✅ Basic multimodal example completed!\n")

def example_embodied_multimodal():
    """Example 2: Embodied agent automatically uses vision+proprio when enabled."""
    print("📖 Example 2: Embodied agent with automatic multimodal features")
    
    # Configure with multimodal enabled
    from emile_mini.config import QSEConfig
    config = QSEConfig()
    config.MULTIMODAL_ENABLED = True
    
    # Create embodied agent and environment
    agent = EmbodiedQSEAgent(config)
    env = EmbodiedEnvironment()
    
    print("Running embodied agent with automatic vision+proprio modalities...")
    
    for step in range(10):
        result = agent.embodied_step(env)
        print(f"Step {step}: position={agent.body.state.position}, energy={agent.body.state.energy:.3f}, context={result.get('context', 0)}")
    
    print("✅ Embodied multimodal example completed!\n")

def example_multimodal_off():
    """Example 3: Demonstrate that default behavior is unchanged."""
    print("📖 Example 3: Default behavior (multimodal disabled)")
    
    # Use default configuration (MULTIMODAL_ENABLED = False)
    from emile_mini.config import QSEConfig
    config = QSEConfig()
    
    agent = EmileAgent(config)
    print(f"MULTIMODAL_ENABLED: {getattr(config, 'MULTIMODAL_ENABLED', False)}")
    
    # Standard agent usage
    for step in range(5):
        result = agent.step()
        print(f"Step {step}: sigma={result['sigma_mean']:.3f}, context={result['context']}")
    
    print("✅ Default behavior example completed!\n")

def example_attention_modes():
    """Example 4: Different attention modes and their effects."""
    print("📖 Example 4: Different attention modes")
    
    CONFIG.MULTIMODAL_ENABLED = True
    agent = EmileAgent(CONFIG)
    
    attention_modes = ['listening', 'reading', 'looking']
    
    for mode in attention_modes:
        print(f"\n--- Testing attention mode: {mode} ---")
        agent.set_attention_mode(mode, steps=10)
        
        # Create appropriate input for each mode
        external_input = {}
        if mode == 'listening':
            external_input['audio'] = np.random.randn(1000) * 0.02
        elif mode == 'reading':
            external_input['text'] = f"This is {mode} mode text input"
        elif mode == 'looking':
            external_input['image'] = np.random.rand(32, 32)
        
        # Get dynamic weights for this mode
        weights = agent._dynamic_weights()
        print(f"Dynamic weights: {weights}")
        
        # Run a few steps
        for i in range(3):
            result = agent.step(external_input=external_input)
            print(f"  Step {i}: attention={agent.attention_mode}, sigma={result['sigma_mean']:.3f}")
    
    print("\n✅ Attention modes example completed!\n")

def example_config_from_yaml():
    """Example 5: Enable multimodal via YAML configuration."""
    print("📖 Example 5: Configuration via YAML")
    
    # Create a sample config file
    config_yaml = """
# Enable multimodal pathway
MULTIMODAL_ENABLED: true
MODALITY_INFLUENCE_SCALE: 0.3

# Standard QSE parameters
K_PSI: 12.0
K_PHI: 6.0
INPUT_COUPLING: 0.3
"""
    
    with open('/tmp/test_config.yaml', 'w') as f:
        f.write(config_yaml)
    
    # Load configuration from YAML
    from emile_mini.config import load_config
    config = load_config('/tmp/test_config.yaml')
    
    print(f"Loaded from YAML - MULTIMODAL_ENABLED: {config.MULTIMODAL_ENABLED}")
    print(f"Loaded from YAML - MODALITY_INFLUENCE_SCALE: {config.MODALITY_INFLUENCE_SCALE}")
    
    # Use the configuration
    agent = EmileAgent(config)
    result = agent.step()
    print(f"Agent created with YAML config, sigma={result['sigma_mean']:.3f}")
    
    print("✅ YAML configuration example completed!\n")

def run_all_examples():
    """Run all usage examples."""
    print("🚀 Multimodal Pathway Usage Examples\n")
    print("These examples show how to use the new opt-in multimodal features.\n")
    
    example_multimodal_off()
    example_basic_multimodal()
    example_embodied_multimodal()
    example_attention_modes()
    example_config_from_yaml()
    
    print("🎉 All examples completed successfully!")
    print("\nKey takeaways:")
    print("- Default behavior is unchanged (MULTIMODAL_ENABLED=False)")
    print("- Enable with CONFIG.MULTIMODAL_ENABLED=True or via YAML")
    print("- Use agent.set_attention_mode(mode, steps) for dynamic attention")
    print("- Embodied agents automatically use vision+proprio when enabled")
    print("- All changes are backward compatible")

if __name__ == "__main__":
    run_all_examples()