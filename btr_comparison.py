"""
BTR Comparison and Analysis Module
Compares BTR algorithms (IQN vs C51) and BTR against SB3 benchmarks.
Generates comparison tables, rankings, and analysis metrics.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path


class BTRComparison:
    """Compare BTR algorithms and results with other methods."""
    
    def __init__(self, btr_results_dir: str = "BTR", sb3_results_dir: str = "SB3"):
        """
        Initialize comparison.
        
        Args:
            btr_results_dir: Directory containing BTR results (CSV/JSON)
            sb3_results_dir: Directory containing SB3 results (CSV/JSON)
        """
        self.btr_results_dir = btr_results_dir
        self.sb3_results_dir = sb3_results_dir
        
        self.btr_results = None
        self.sb3_results = None
        self.comparison_df = None
    
    def load_btr_results(self) -> pd.DataFrame:
        """Load BTR results from CSV."""
        csv_path = os.path.join(self.btr_results_dir, "btr_results.csv")
        if os.path.exists(csv_path):
            self.btr_results = pd.read_csv(csv_path)
            return self.btr_results
        else:
            print(f"Warning: BTR results not found at {csv_path}")
            return None
    
    def load_sb3_results(self) -> pd.DataFrame:
        """Load SB3 results from CSV."""
        csv_path = os.path.join(self.sb3_results_dir, "sb3_results.csv")
        if os.path.exists(csv_path):
            self.sb3_results = pd.read_csv(csv_path)
            return self.sb3_results
        else:
            print(f"Warning: SB3 results not found at {csv_path}")
            return None
    
    def compare_within_btr(self) -> pd.DataFrame:
        """
        Compare BTR algorithms (IQN vs C51) within themselves.
        
        Returns:
            DataFrame with comparison statistics
        """
        if self.btr_results is None:
            self.load_btr_results()
        
        if self.btr_results is None or len(self.btr_results) == 0:
            print("No BTR results to compare")
            return None
        
        comparison = []
        for algo in self.btr_results["algorithm"].unique():
            algo_data = self.btr_results[self.btr_results["algorithm"] == algo]
            
            comparison.append({
                "algorithm": algo,
                "n_seeds": len(algo_data),
                "reward_mean": algo_data["reward_mean"].mean(),
                "reward_std": algo_data["reward_mean"].std(),
                "price_mean": algo_data["price_mean"].mean(),
                "price_std": algo_data["price_mean"].std(),
                "best_price": algo_data["price_mean"].min(),
                "worst_price": algo_data["price_mean"].max(),
                "price_range": algo_data["price_mean"].max() - algo_data["price_mean"].min(),
            })
        
        comparison_df = pd.DataFrame(comparison)
        comparison_df = comparison_df.sort_values("price_mean")
        
        return comparison_df
    
    def compare_btr_vs_sb3(self) -> pd.DataFrame:
        """
        Compare BTR algorithms with SB3 benchmarks.
        
        Returns:
            DataFrame with combined comparison
        """
        btr_comp = self.compare_within_btr()
        sb3_data = self.load_sb3_results()
        
        if btr_comp is None or sb3_data is None:
            print("Cannot compare: missing BTR or SB3 results")
            return None
        
        # Prepare SB3 data
        sb3_comp = []
        for algo in sb3_data["algorithm"].unique():
            algo_data = sb3_data[sb3_data["algorithm"] == algo]
            sb3_comp.append({
                "algorithm": f"SB3_{algo}",
                "n_seeds": len(algo_data),
                "reward_mean": algo_data["reward_mean"].mean(),
                "reward_std": algo_data["reward_mean"].std(),
                "price_mean": algo_data["price_mean"].mean(),
                "price_std": algo_data["price_mean"].std(),
                "best_price": algo_data["price_mean"].min(),
                "worst_price": algo_data["price_mean"].max(),
                "price_range": algo_data["price_mean"].max() - algo_data["price_mean"].min(),
            })
        
        sb3_df = pd.DataFrame(sb3_comp)
        
        # Combine
        combined_df = pd.concat([btr_comp, sb3_df], ignore_index=True)
        combined_df = combined_df.sort_values("price_mean")
        combined_df["price_rank"] = range(1, len(combined_df) + 1)
        
        self.comparison_df = combined_df
        return combined_df
    
    def get_best_agent(self, criterion: str = "price_mean") -> Dict[str, Any]:
        """
        Get best agent across all results.
        
        Args:
            criterion: Metric to rank by ("price_mean", "reward_mean", etc.)
            
        Returns:
            Dictionary with best agent info
        """
        if self.btr_results is None:
            self.load_btr_results()
        
        if self.btr_results is None or len(self.btr_results) == 0:
            return None
        
        best_idx = self.btr_results[criterion].idxmin()
        best_agent = self.btr_results.loc[best_idx].to_dict()
        
        return best_agent
    
    def get_worst_agent(self, criterion: str = "price_mean") -> Dict[str, Any]:
        """Get worst agent by criterion."""
        if self.btr_results is None:
            self.load_btr_results()
        
        if self.btr_results is None or len(self.btr_results) == 0:
            return None
        
        worst_idx = self.btr_results[criterion].idxmax()
        worst_agent = self.btr_results.loc[worst_idx].to_dict()
        
        return worst_agent
    
    def get_algorithm_statistics(self, algorithm: str) -> Dict[str, float]:
        """Get statistics for a specific algorithm."""
        if self.btr_results is None:
            self.load_btr_results()
        
        algo_data = self.btr_results[self.btr_results["algorithm"] == algorithm]
        
        if len(algo_data) == 0:
            return None
        
        stats = {
            "algorithm": algorithm,
            "n_seeds": len(algo_data),
            "reward_mean": float(algo_data["reward_mean"].mean()),
            "reward_min": float(algo_data["reward_mean"].min()),
            "reward_max": float(algo_data["reward_mean"].max()),
            "reward_std": float(algo_data["reward_mean"].std()),
            "price_mean": float(algo_data["price_mean"].mean()),
            "price_min": float(algo_data["price_mean"].min()),
            "price_max": float(algo_data["price_mean"].max()),
            "price_std": float(algo_data["price_mean"].std()),
        }
        
        return stats
    
    def generate_comparison_report(self, output_path: str = "BTR/comparison_report.md"):
        """Generate markdown comparison report."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        report = "# BTR Algorithm Comparison Report\n\n"
        
        # Within-BTR comparison
        btr_comp = self.compare_within_btr()
        if btr_comp is not None:
            report += "## Within-BTR Comparison (IQN vs C51)\n\n"
            report += btr_comp.to_markdown(index=False)
            report += "\n\n"
        
        # BTR vs SB3 comparison
        combined = self.compare_btr_vs_sb3()
        if combined is not None:
            report += "## BTR vs SB3 Comparison\n\n"
            report += combined.to_markdown(index=False)
            report += "\n\n"
            
            # Best agent
            best = self.get_best_agent()
            if best:
                report += "## Best Agent\n\n"
                report += f"- **Algorithm**: {best['algorithm']}\n"
                report += f"- **Seed**: {best['seed']}\n"
                report += f"- **Price Mean**: {best['price_mean']:.2f}\n"
                report += f"- **Price Std**: {best.get('price_std', 'N/A')}\n"
                report += f"- **Reward Mean**: {best['reward_mean']:.2f}\n"
                report += f"- **Model Path**: {best.get('model_path', 'N/A')}\n\n"
        
        # Statistics
        report += "## Detailed Statistics\n\n"
        for algo in self.btr_results["algorithm"].unique():
            stats = self.get_algorithm_statistics(algo)
            if stats:
                report += f"### {algo}\n\n"
                report += f"- **Seeds**: {stats['n_seeds']}\n"
                report += f"- **Reward**: {stats['reward_mean']:.2f} ± {stats['reward_std']:.2f} (min: {stats['reward_min']:.2f}, max: {stats['reward_max']:.2f})\n"
                report += f"- **Price**: {stats['price_mean']:.2f} ± {stats['price_std']:.2f} (min: {stats['price_min']:.2f}, max: {stats['price_max']:.2f})\n\n"
        
        with open(output_path, "w") as f:
            f.write(report)
        
        print(f"Comparison report saved to: {output_path}")
        
        return report
    
    def save_comparison_csv(self, output_path: str = "BTR/comparison_all.csv"):
        """Save combined comparison to CSV."""
        combined = self.compare_btr_vs_sb3()
        if combined is not None:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            combined.to_csv(output_path, index=False)
            print(f"Comparison CSV saved to: {output_path}")
            return combined
        return None
    
    def save_best_agent_info(self, output_path: str = "BTR/best_agent/summary.json"):
        """Save best agent information."""
        best = self.get_best_agent()
        if best:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(best, f, indent=2)
            print(f"Best agent info saved to: {output_path}")
            return best
        return None


def compare_algorithm_variants(
    results_df: pd.DataFrame,
    algorithm: str,
    metric: str = "price_mean"
) -> Dict[str, List[float]]:
    """
    Compare variants of an algorithm across seeds.
    
    Args:
        results_df: Results DataFrame
        algorithm: Algorithm name
        metric: Metric to compare
        
    Returns:
        Dictionary of metric values per seed
    """
    algo_data = results_df[results_df["algorithm"] == algorithm]
    return {
        "values": algo_data[metric].tolist(),
        "mean": float(algo_data[metric].mean()),
        "std": float(algo_data[metric].std()),
        "min": float(algo_data[metric].min()),
        "max": float(algo_data[metric].max()),
    }


def compute_improvement(baseline: float, improved: float, higher_is_better: bool = False) -> float:
    """
    Compute percent improvement.
    
    Args:
        baseline: Baseline value
        improved: Improved value
        higher_is_better: If True, higher values are better; else lower is better
        
    Returns:
        Percent improvement
    """
    if baseline == 0:
        return 0.0
    
    if higher_is_better:
        return 100 * (improved - baseline) / abs(baseline)
    else:
        return 100 * (baseline - improved) / abs(baseline)


def rank_algorithms(
    results_df: pd.DataFrame,
    metric: str = "price_mean",
    ascending: bool = True
) -> pd.DataFrame:
    """
    Rank algorithms by metric.
    
    Args:
        results_df: Results DataFrame
        metric: Metric to rank by
        ascending: If True, lower values rank higher
        
    Returns:
        DataFrame with ranks
    """
    ranked = results_df.copy()
    ranked["rank"] = ranked[metric].rank(ascending=ascending)
    ranked = ranked.sort_values("rank")
    
    return ranked
