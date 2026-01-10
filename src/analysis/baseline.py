from dataclasses import dataclass, field
from typing import List, Dict
from datetime import datetime

from ..storage import ResultsStorage, TestRun
from .statistics import calculate_statistics, Statistics


@dataclass
class BaselineMetrics:
    """
    Baseline performance metrics for a model.
    
    This captures "normal" performance that we compare against later.
    """
    model_name: str
    start_date: str
    end_date: str
    num_runs: int
    
    # Overall metrics
    overall_stats: Statistics
    
    # By category
    by_category: Dict[str, Statistics] = field(default_factory=dict)
    
    # By test
    by_test: Dict[str, Statistics] = field(default_factory=dict)
    
    # Latency
    latency_stats: Statistics = None
    
    def __repr__(self) -> str:
        return (
            f"BaselineMetrics("
            f"model={self.model_name}, "
            f"runs={self.num_runs}, "
            f"mean_score={self.overall_stats.mean:.1%})"
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "num_runs": self.num_runs,
            "overall_mean": self.overall_stats.mean,
            "overall_std": self.overall_stats.std,
            "overall_ci": self.overall_stats.confidence_interval_95,
            "by_category": {
                cat: {"mean": stats.mean, "std": stats.std}
                for cat, stats in self.by_category.items()
            },
            "latency_ms": {
                "mean": self.latency_stats.mean,
                "std": self.latency_stats.std,
            } if self.latency_stats else None,
        }


def calculate_baseline(
    storage: ResultsStorage,
    model_name: str,
    start_date: str = None,
    end_date: str = None,
    num_runs: int = 7
) -> BaselineMetrics:
    """
    Calculate baseline metrics for a model.
    
    Args:
        storage: ResultsStorage instance
        model_name: Which model to analyze
        start_date: Optional start date (YYYY-MM-DD)
        end_date: Optional end date (YYYY-MM-DD)
        num_runs: If dates not specified, use first N runs
    
    Returns:
        BaselineMetrics with all statistics
    
    Example:
        storage = ResultsStorage()
        baseline = calculate_baseline(storage, "GPT-4 Turbo", num_runs=7)
        print(f"Baseline score: {baseline.overall_stats.mean:.1%}")
    """
    if start_date and end_date:
        runs = storage.load_runs_since(start_date)
        runs = [r for r in runs if r.timestamp[:10] <= end_date]
    else:
        all_runs = storage.load_all_runs()
        runs = []
        for run in all_runs:
            if model_name in run.models_tested:
                runs.append(run)
                if len(runs) >= num_runs:
                    break
    
    if not runs:
        raise ValueError("No runs found for baseline calculation")
    
    all_results = []
    for run in runs:
        model_results = [r for r in run.results if r.model_name == model_name]
        all_results.extend(model_results)
    
    if not all_results:
        raise ValueError(f"No results found for model: {model_name}")
    
    all_scores = [r.score for r in all_results]
    overall_stats = calculate_statistics(all_scores)
    
    by_category = {}
    categories = set(r.category for r in all_results if r.category)
    for category in categories:
        cat_results = [r for r in all_results if r.category == category]
        cat_scores = [r.score for r in cat_results]
        if cat_scores:
            by_category[category] = calculate_statistics(cat_scores)
    
    by_test = {}
    test_ids = set(r.test_id for r in all_results)
    for test_id in test_ids:
        test_results = [r for r in all_results if r.test_id == test_id]
        test_scores = [r.score for r in test_results]
        if test_scores:
            by_test[test_id] = calculate_statistics(test_scores)
    
    latencies = [r.latency_ms for r in all_results if r.success]
    latency_stats = calculate_statistics(latencies) if latencies else None
    
    return BaselineMetrics(
        model_name=model_name,
        start_date=runs[0].timestamp[:10],
        end_date=runs[-1].timestamp[:10],
        num_runs=len(runs),
        overall_stats=overall_stats,
        by_category=by_category,
        by_test=by_test,
        latency_stats=latency_stats,
    )


def get_all_baselines(
    storage: ResultsStorage,
    num_runs: int = 7
) -> Dict[str, BaselineMetrics]:
    """
    Calculate baselines for all models.
    
    Args:
        storage: ResultsStorage instance
        num_runs: Number of runs to use for baseline
    
    Returns:
        Dictionary mapping model names to BaselineMetrics
    """
    runs = storage.load_all_runs()
    
    if not runs:
        raise ValueError("No runs found")
    
    model_names = set()
    for run in runs:
        model_names.update(run.models_tested)
    
    baselines = {}
    for model_name in model_names:
        try:
            baseline = calculate_baseline(storage, model_name, num_runs=num_runs)
            baselines[model_name] = baseline
        except Exception as e:
            print(f"Warning: Could not calculate baseline for {model_name}: {e}")
    
    return baselines


def print_baseline_report(baseline: BaselineMetrics):
    """
    Print a human-readable baseline report.
    
    Args:
        baseline: BaselineMetrics to display
    """
    print(f"\n{'='*60}")
    print(f"BASELINE REPORT: {baseline.model_name}")
    print(f"{'='*60}")
    print(f"Period: {baseline.start_date} to {baseline.end_date}")
    print(f"Runs: {baseline.num_runs}")
    print()
    
    print("Overall Performance:")
    stats = baseline.overall_stats
    ci_lower, ci_upper = stats.confidence_interval_95
    print(f"  Mean score: {stats.mean:.1%}")
    print(f"  Std dev: {stats.std:.3f}")
    print(f"  95% CI: [{ci_lower:.1%}, {ci_upper:.1%}]")
    print(f"  Range: [{stats.min:.1%}, {stats.max:.1%}]")
    print()
    
    print("By Category:")
    for category, cat_stats in sorted(baseline.by_category.items()):
        print(f"  {category:15s}: {cat_stats.mean:.1%} ± {cat_stats.std:.3f}")
    print()
    
    if baseline.latency_stats:
        print("Performance:")
        print(f"  Avg latency: {baseline.latency_stats.mean:.0f}ms")
        print(f"  Latency std: {baseline.latency_stats.std:.0f}ms")


# Test the baseline calculation
if __name__ == "__main__":
    from ..storage import ResultsStorage
    
    print("Testing baseline calculation...")
    
    storage = ResultsStorage()
    runs = storage.load_all_runs()
    
    if len(runs) < 2:
        print("Need at least 2 runs to calculate baseline")
        print("Run the test suite a few more times first.")
    else:
        print(f"\nFound {len(runs)} runs")
        print(f"Calculating baseline from first {min(7, len(runs))} runs...")
        
        baselines = get_all_baselines(storage, num_runs=min(7, len(runs)))
        
        for model_name, baseline in baselines.items():
            print_baseline_report(baseline)