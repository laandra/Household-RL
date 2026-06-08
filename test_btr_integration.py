"""
Quick integration test for BTR modules.
Validates that all modules can be imported and basic training works.
"""

import numpy as np
import torch
from Environment import HouseholdEnvironment
from btr_environment_adapter import BTREnvironmentAdapter
from btr_agent import BTRAgent
from btr_config_builder import get_btr_algorithm_config

def test_imports():
    """Test that all BTR modules can be imported."""
    print("Testing imports...")
    try:
        from btr_runner import BTRTrainer, run_btr_benchmark
        from btr_comparison import BTRComparison
        from btr_quickstart import (
            quickstart_single_seed_training,
            quickstart_multi_seed_benchmark,
            quickstart_comparison,
        )
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_environment_adapter():
    """Test BTREnvironmentAdapter with a simple dummy environment."""
    print("\nTesting BTREnvironmentAdapter...")
    try:
        # Create simple test environment
        class DummyEnv:
            def __init__(self):
                from gymnasium import spaces
                self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
                self.action_space = spaces.Discrete(5)
            
            def reset(self, seed=None):
                return np.zeros(4, dtype=np.float32), {}
            
            def step(self, action):
                return (
                    np.zeros(4, dtype=np.float32),
                    1.0,
                    False,
                    False,
                    {"cumulative_payment": 0.0}
                )
        
        dummy_env = DummyEnv()
        adapter = BTREnvironmentAdapter(dummy_env, device="cpu", use_torch=False)
        
        # Test reset
        state, info = adapter.reset(seed=42)
        assert isinstance(state, np.ndarray), f"Expected numpy array, got {type(state)}"
        assert state.dtype == np.float32, f"Expected float32, got {state.dtype}"
        assert state.shape == (4,), f"Expected shape (4,), got {state.shape}"
        
        # Test step
        state, reward, terminated, truncated, info = adapter.step(0)
        assert isinstance(state, np.ndarray), f"Expected numpy array, got {type(state)}"
        assert isinstance(reward, float), f"Expected float, got {type(reward)}"
        
        print("✓ BTREnvironmentAdapter working correctly")
        return True
    except Exception as e:
        print(f"✗ BTREnvironmentAdapter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_creation():
    """Test agent creation with different algorithms."""
    print("\nTesting BTRAgent creation...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        for algo in ["IQN", "C51"]:
            config = get_btr_algorithm_config(algo, enable_optuna=False)
            params = config.get_params()
            
            agent = BTRAgent(
                state_shape=4,
                n_actions=5,
                algorithm=algo,
                device=device,
                **params
            )
            
            print(f"  ✓ {algo} agent created successfully")
        
        print("✓ Agent creation working for all algorithms")
        return True
    except Exception as e:
        print(f"✗ Agent creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_action_selection():
    """Test that agents can select actions."""
    print("\nTesting action selection...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        for algo in ["IQN", "C51"]:
            config = get_btr_algorithm_config(algo, enable_optuna=False)
            params = config.get_params()
            
            agent = BTRAgent(
                state_shape=4,
                n_actions=5,
                algorithm=algo,
                device=device,
                **params
            )
            
            # Test action selection
            state = np.zeros(4, dtype=np.float32)
            action = agent.select_action(state, training=True)
            
            assert isinstance(action, int), f"Expected int, got {type(action)}"
            assert 0 <= action < 5, f"Action out of bounds: {action}"
            
            print(f"  ✓ {algo} action selection: {action}")
        
        print("✓ Action selection working for all algorithms")
        return True
    except Exception as e:
        print(f"✗ Action selection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_learning():
    """Test that agent can perform a learning step."""
    print("\nTesting batch learning...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        config = get_btr_algorithm_config("IQN", enable_optuna=False)
        params = config.get_params()
        params["learning_starts"] = 2  # Only need 2 samples to start learning
        params["batch_size"] = 2
        
        agent = BTRAgent(
            state_shape=4,
            n_actions=5,
            algorithm="IQN",
            device=device,
            **params
        )
        
        # Add some transitions
        state = np.zeros(4, dtype=np.float32)
        next_state = np.zeros(4, dtype=np.float32)
        
        for _ in range(5):
            agent.store_transition(state, 0, 1.0, next_state, False)
        
        # Try learning step
        loss = agent.learn_step()
        if loss is not None:
            print(f"  ✓ Learning step completed, loss: {loss:.4f}")
        else:
            print(f"  ✓ Learning step (warmup phase)")
        
        print("✓ Batch learning working")
        return True
    except Exception as e:
        print(f"✗ Batch learning failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("BTR Integration Test Suite")
    print("="*60)
    
    results = {
        "Imports": test_imports(),
        "Environment Adapter": test_environment_adapter(),
        "Agent Creation": test_agent_creation(),
        "Action Selection": test_action_selection(),
        "Batch Learning": test_batch_learning(),
    }
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! BTR implementation is ready.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. See errors above.")
        return 1


if __name__ == "__main__":
    exit(main())
