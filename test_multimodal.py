#!/usr/bin/env python3
"""
Simple test suite for multimodal pathway functionality.
Tests backward compatibility and new multimodal features.
"""

import numpy as np
from emile_mini.agent import EmileAgent
from emile_mini.config import QSEConfig
from emile_mini.embodied_qse_emile import EmbodiedQSEAgent, EmbodiedEnvironment
from emile_mini.multimodal import ModalityFeature, TextAdapter, ImageAdapter, AudioAdapter, ModalityAttentionPolicy

def test_backward_compatibility():
    """Test that existing functionality works unchanged by default."""
    print("🧪 Testing backward compatibility...")
    
    # Test 1: Default config should have MULTIMODAL_ENABLED = False
    config = QSEConfig()
    assert getattr(config, 'MULTIMODAL_ENABLED', False) == False, "Default MULTIMODAL_ENABLED should be False"
    assert getattr(config, 'MODALITY_INFLUENCE_SCALE', 0.25) == 0.25, "Default MODALITY_INFLUENCE_SCALE should be 0.25"
    
    # Test 2: Agent creation should work as before
    agent = EmileAgent(config)
    assert agent is not None, "Agent creation should work"
    
    # Test 3: Agent step should work as before
    result = agent.step()
    assert isinstance(result, dict), "Step should return dict"
    assert 'sigma_mean' in result, "Result should contain sigma_mean"
    
    # Test 4: Embodied agent should work as before
    embodied_agent = EmbodiedQSEAgent(config)
    env = EmbodiedEnvironment()
    embodied_result = embodied_agent.embodied_step(env)
    assert isinstance(embodied_result, dict), "Embodied step should return dict"
    
    print("✅ Backward compatibility tests passed!")

def test_multimodal_functionality():
    """Test new multimodal functionality when enabled."""
    print("🧪 Testing multimodal functionality...")
    
    # Test 1: Config with multimodal enabled
    config = QSEConfig()
    config.MULTIMODAL_ENABLED = True
    
    # Test 2: Agent with multimodal enabled
    agent = EmileAgent(config)
    assert agent.mm_text is not None, "Text adapter should be initialized"
    assert agent.mm_image is not None, "Image adapter should be initialized"
    assert agent.mm_audio is not None, "Audio adapter should be initialized"
    assert agent._attention is not None, "Attention policy should be initialized"
    
    # Test 3: Attention mode API
    agent.set_attention_mode('listening', steps=5)
    assert agent.attention_mode == 'listening', "Attention mode should be set"
    assert agent._attention_steps_left == 5, "Attention steps should be set"
    
    # Test 4: Attention tick-down
    agent.step()
    assert agent._attention_steps_left == 4, "Attention steps should decrease"
    
    # Test 5: Multimodal input processing
    external_input = {
        'text': 'hello world',
        'audio': np.random.randn(1000) * 0.01,
        'image': np.random.rand(16, 16)
    }
    result = agent.step(external_input=external_input)
    assert isinstance(result, dict), "Step with multimodal input should work"
    
    print("✅ Multimodal functionality tests passed!")

def test_adapters():
    """Test individual adapter functionality."""
    print("🧪 Testing adapters...")
    
    # Test 1: Text adapter
    text_adapter = TextAdapter(dim=32)
    text_vec = text_adapter.encode("hello world test")
    assert text_vec.shape == (32,), f"Text vector shape should be (32,), got {text_vec.shape}"
    assert np.all(np.isfinite(text_vec)), "Text vector should have finite values"
    
    # Test 2: Image adapter
    image_adapter = ImageAdapter(dim=64)
    test_image = np.random.rand(8, 8)
    image_vec = image_adapter.encode(test_image)
    assert image_vec.shape == (64,), f"Image vector shape should be (64,), got {image_vec.shape}"
    assert np.all(np.isfinite(image_vec)), "Image vector should have finite values"
    
    # Test 3: Audio adapter
    audio_adapter = AudioAdapter(dim=32)
    test_audio = np.random.randn(1000) * 0.1
    audio_vec = audio_adapter.encode(test_audio)
    assert audio_vec.shape == (32,), f"Audio vector shape should be (32,), got {audio_vec.shape}"
    assert np.all(np.isfinite(audio_vec)), "Audio vector should have finite values"
    
    print("✅ Adapter tests passed!")

def test_attention_policy():
    """Test attention policy functionality."""
    print("🧪 Testing attention policy...")
    
    policy = ModalityAttentionPolicy()
    
    # Mock agent with attention mode
    class MockAgent:
        def __init__(self):
            self.attention_mode = 'listening'
            self.context = MockContext()
    
    class MockContext:
        def get_current(self):
            return 1
    
    agent = MockAgent()
    weights = policy.weights_for(agent)
    
    assert isinstance(weights, dict), "Weights should be a dict"
    assert 'audio' in weights, "Audio weight should be present"
    assert weights['audio'] > weights['text'], "Audio should be weighted higher in listening mode"
    
    print("✅ Attention policy tests passed!")

def test_cli_compatibility():
    """Test that CLI still works with new functionality."""
    print("🧪 Testing CLI compatibility...")
    
    # Import and check that CLI can still be imported
    try:
        from emile_mini.cli import main
        print("✅ CLI import successful!")
    except Exception as e:
        assert False, f"CLI import failed: {e}"

def test_embodied_multimodal():
    """Test embodied agent with multimodal features."""
    print("🧪 Testing embodied multimodal...")
    
    # Test with multimodal enabled
    config = QSEConfig()
    config.MULTIMODAL_ENABLED = True
    
    agent = EmbodiedQSEAgent(config)
    env = EmbodiedEnvironment()
    
    # Should work without errors
    result = agent.embodied_step(env)
    assert isinstance(result, dict), "Embodied step with multimodal should return dict"
    
    print("✅ Embodied multimodal tests passed!")

def run_all_tests():
    """Run all tests."""
    print("🚀 Running multimodal pathway test suite...\n")
    
    try:
        test_backward_compatibility()
        test_multimodal_functionality()
        test_adapters()
        test_attention_policy()
        test_cli_compatibility()
        test_embodied_multimodal()
        
        print("\n🎉 All tests passed! Multimodal pathway implementation successful.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)