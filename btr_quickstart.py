"""
BTR Quick Start Example
Demonstrates basic BTR training, evaluation, and comparison with a household environment.
"""

import os
import sys
import numpy as np
import torch
from typing import Optional

# Assuming Environment.py is in the same directory
from Environment import HouseholdEnvironment
from btr_runner import BTRTrainer, run_btr_benchmark
from btr_comparison import BTRComparison


def create_household_environment(
    dataset,
    dataset_norm: Optional[np.ndarray] = None,
    observation_mode: str = "compact",
    episode_length: Optional[int] = None,
    **kwargs
) -> HouseholdEnvironment:
    """
    Create a household environment instance.
    
    Args:
        dataset: DataFrame with household data
        dataset_norm: Normalized dataset (or None to use dataset)
        observation_mode: "compact" or "sliding_window"
        episode_length: Length of episodes (or None for full dataset)
        **kwargs: Additional environment parameters
        
    Returns:
        HouseholdEnvironment instance
    """
    return HouseholdEnvironment(
        dataset=dataset,
        dataset_norm=dataset_norm,
        observation_mode=observation_mode,
        episode_length=episode_length,
        **kwargs
    )


def quickstart_single_seed_training(
    train_env,
    eval_env,
    algorithm: str = "IQN",
    total_timesteps: int = 10000,
    use_optuna: bool = False,
    device: str = "cpu",
    verbose: int = 1,
) -> dict:
    """
    Quick example: Train a single seed of IQN or C51.
    
    Args:
        train_env: Training environment
        eval_env: Evaluation environment
        algorithm: "IQN" or "C51"
        total_timesteps: Training timesteps
        use_optuna: Enable Optuna optimization
        device: PyTorch device
        verbose: Verbosity
        
    Returns:
        Training metrics dictionary
    """
    print(f"\n{'='*60}")
    print(f"BTR Quick Start - Single Seed Training ({algorithm})")
    print(f"{'='*60}")
    
    trainer = BTRTrainer(
        algorithm_name=algorithm,
        env=train_env,
        eval_env=eval_env,
        total_timesteps=total_timesteps,
        n_eval_episodes=5,
        eval_freq=2000,
        seed=42,
        device=device,
        verbose=verbose,
    )
    
    metrics = trainer.train(use_optuna=use_optuna, n_optuna_trials=2)
    
    if metrics["success"]:
        print(f"\n✓ Training successful!")
        print(f"  Reward (checkpoint mean): {metrics['reward_mean']:.2f} ± {metrics['reward_std']:.2f}")
        print(f"  Energy Cost (checkpoint mean): {metrics['price_mean']:.2f} ± {metrics['price_std']:.2f}")
        
        # Save model
        model_path = f"quickstart_{algorithm}_model.pt"
        trainer.save_agent(model_path)
        print(f"  Model saved to: {model_path}")
    else:
        print(f"✗ Training failed: {metrics.get('error', 'Unknown error')}")
    
    return metrics


def quickstart_multi_seed_benchmark(
    train_env,
    eval_env,
    test_env,
    total_timesteps: int = 10000,
    n_seeds: int = 2,
    use_optuna: bool = False,
    device: str = "cpu",
    output_dir: str = "BTR_quickstart",
    verbose: int = 1,
) -> dict:
    """
    Quick example: Multi-seed benchmark comparing IQN and C51.
    
    Args:
        train_env: Training environment
        eval_env: Evaluation environment
        test_env: Test environment
        total_timesteps: Training timesteps per seed
        n_seeds: Number of seeds
        use_optuna: Enable optimization
        device: PyTorch device
        output_dir: Output directory
        verbose: Verbosity
        
    Returns:
        Benchmark results dictionary
    """
    print(f"\n{'='*60}")
    print(f"BTR Quick Start - Multi-Seed Benchmark")
    print(f"{'='*60}")
    
    results = run_btr_benchmark(
        train_env=train_env,
        eval_env=eval_env,
        test_env=test_env,
        algorithms=["IQN", "C51"],
        total_timesteps=total_timesteps,
        n_seeds=n_seeds,
        n_eval_episodes=5,
        output_dir=output_dir,
        device=device,
        use_optuna=use_optuna,
        n_optuna_trials=2,
        verbose=verbose,
    )
    
    print(f"\n{'='*60}")
    print("Benchmark Results Summary")
    print(f"{'='*60}")
    for algo, metrics in results.items():
        print(f"\n{algo}:")
        print(f"  Reward: {metrics['reward_mean']:.2f} ± {metrics['reward_std']:.2f}")
        print(f"  Energy Cost: {metrics['price_mean']:.2f} ± {metrics['price_std']:.2f}")
        if "reward_7day" in metrics and "price_7day" in metrics:
            print(f"  Quick 7-day Reward: {metrics['reward_7day']:.2f}")
            print(f"  Quick 7-day Energy Cost: {metrics['price_7day']:.2f}")
    
    return results


def quickstart_comparison(btr_dir: str = "BTR_quickstart", sb3_dir: str = "SB3") -> None:
    """
    Quick example: Compare BTR with SB3 results.
    
    Args:
        btr_dir: BTR results directory
        sb3_dir: SB3 results directory
    """
    print(f"\n{'='*60}")
    print("BTR vs SB3 Comparison")
    print(f"{'='*60}")
    
    comparator = BTRComparison(btr_results_dir=btr_dir, sb3_results_dir=sb3_dir)
    
    # Load and compare
    btr_data = comparator.load_btr_results()
    sb3_data = comparator.load_sb3_results()
    
    if btr_data is not None:
        print("\nBTR Results Loaded:")
        print(btr_data.to_string())
    
    if sb3_data is not None:
        print("\nSB3 Results Loaded:")
        print(sb3_data.to_string())
    
    # Generate report
    report = comparator.generate_comparison_report(
        output_path=os.path.join(btr_dir, "comparison_report.md")
    )
    
    # Save comparison CSV
    comparator.save_comparison_csv(
        output_path=os.path.join(btr_dir, "comparison_all.csv")
    )
    
    # Get best agent
    best = comparator.get_best_agent()
    if best:
        print("\n✓ Best Agent Found:")
        print(f"  Algorithm: {best['algorithm']}")
        print(f"  Seed: {best['seed']}")
        print(f"  Energy Cost: {best['price_mean']:.2f}")


if __name__ == "__main__":
    """
    Example usage: Initialize household environment and run BTR training.
    
    Note: In practice, you would load your actual household data here.
    This is a template showing the API.
    """
    
    print("BTR Quick Start Template")
    print("="*60)
    print("\nThis script demonstrates how to use BTR with a household environment.")
    print("\nUsage steps:")
    print("1. Load your household dataset (VhodniPodatki.csv or similar)")
    print("2. Create train_env and eval_env instances")
    print("3. Call quickstart_single_seed_training() or quickstart_multi_seed_benchmark()")
    print("4. Use quickstart_comparison() to compare results")
    print("\nExample:")
    print("""
    
    # Load data
    import pandas as pd
    data = pd.read_csv("VhodniPodatki.csv", index_col=0, parse_dates=True)
    
    # Create environments
    train_env = create_household_environment(
        dataset=data[:1000],
        observation_mode="compact"
    )
    eval_env = create_household_environment(
        dataset=data[1000:2000],
        observation_mode="compact"
    )
    test_env = create_household_environment(
        dataset=data[2000:],
        observation_mode="compact"
    )
    
    # Train single IQN agent
    metrics = quickstart_single_seed_training(
        train_env=train_env,
        eval_env=eval_env,
        algorithm="IQN",
        total_timesteps=100000,
        device="cpu"
    )
    
    # Or run multi-seed benchmark
    results = quickstart_multi_seed_benchmark(
        train_env=train_env,
        eval_env=eval_env,
        test_env=test_env,
        total_timesteps=100000,
        n_seeds=3,
        device="cpu"
    )
    
    # Compare with SB3
    quickstart_comparison(btr_dir="BTR_quickstart", sb3_dir="SB3")
    
    """)
    
    print("\nFor detailed examples, see smart_home_nanogrid_DQN.ipynb")
    print("="*60)
