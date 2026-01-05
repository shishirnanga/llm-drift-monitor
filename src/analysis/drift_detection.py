"""
src/analysis/drift_detection.py

Detect when model performance has drifted significantly.

This is the core of the drift monitor - comparing current performance
to baseline and flagging when changes are statistically significant.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

from ..storage import ResultsStorage, TestRun
from .baseline import BaselineMetrics, calculate_baseline
from .statistics import (
    calculate_statistics,
    welch_ttest,
    cohens_d,
    interpret_cohens_d,
    interpret_pvalue,
    Statistics,
)


class DriftSeverity(Enum):
    """How severe is the drift?"""
    NONE = "none"           # No significant drift
    MINOR = "minor"         # Significant but small effect
    MODERATE = "moderate"   # Significant with medium effect
    MAJOR = "major"         # Significant with large effect


@dataclass
class DriftResult:
    """
    Result of drift detection analysis.
    
    Contains all the information about whether drift was detected,
    how severe it is, and the supporting statistics.
    """
    model_name: str
    test_period: str
    
    # Was drift detected?
    drift_detected: bool
    severity: DriftSeverity
    
    # Statistics
    baseline_mean: float
    current_mean: float
    change_percent: float  # (current - baseline) / baseline * 100
    
    # Statistical tests
    p_value: float
    cohens_d: float
    
    # By category
    category_drift: Dict[str, bool] = None
    
    # Human-readable summary
    summary: str = ""
    
    def __repr__(self) -> str:
        status = "🚨 DRIFT" if self.drift_detected else "✅ STABLE"
        return (
            f"{status} {self.model_name}: "
            f"{self.baseline_mean:.1%} → {self.current_mean:.1%} "
            f"({self.change_percent:+.1f}%) "
            f"p={self.p_value:.3f}, d={self.cohens_d:.2f}"
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "test_period": self.test_period,
            "drift_detected": self.drift_detected,
            "severity": self.severity.value,
            "baseline_mean": self.baseline_mean,
            "current_mean": self.current_mean,
            "change_percent": self.change_percent,
            "p_value": self.p_value,
            "cohens_d": self.cohens_d,
            "summary": self.summary,
        }


def detect_drift(
    storage: ResultsStorage,
    model_name: str,
    baseline_runs: int = 7,
    current_runs: int = 3,
    significance_level: float = 0.05
) -> DriftResult:
    """
    Detect if model performance has drifted from baseline.
    
    Args:
        storage: ResultsStorage instance
        model_name: Which model to analyze
        baseline_runs: Number of runs to use for baseline (default: first 7)
        current_runs: Number of recent runs to test (default: last 3)
        significance_level: p-value threshold for significance (default: 0.05)
    
    Returns:
        DriftResult with detection outcome and statistics
    
    Example:
        storage = ResultsStorage()
        result = detect_drift(storage, "GPT-4 Turbo")
        
        if result.drift_detected:
            print(f"⚠️ Drift detected! {result.summary}")
        else:
            print("✅ No significant drift")
    """
    all_runs = storage.load_all_runs()
    
    if len(all_runs) < baseline_runs + current_runs:
        raise ValueError(
            f"Need at least {baseline_runs + current_runs} runs "
            f"({baseline_runs} baseline + {current_runs} current). "
            f"Only have {len(all_runs)}."
        )
    
    # Calculate baseline from first N runs
    baseline = calculate_baseline(storage, model_name, num_runs=baseline_runs)
    
    # Get current results from last M runs
    current_runs_data = all_runs[-current_runs:]
    current_results = []
    for run in current_runs_data:
        model_results = [r for r in run.results if r.model_name == model_name]
        current_results.extend(model_results)
    
    if not current_results:
        raise ValueError(f"No current results found for {model_name}")
    
    # Extract scores
    baseline_scores = []
    baseline_run_count = 0
    for run in all_runs:
        if model_name in run.models_tested:
            model_results = [r for r in run.results if r.model_name == model_name]
            baseline_scores.extend([r.score for r in model_results])
            baseline_run_count += 1
            if baseline_run_count >= baseline_runs:
                break

    current_scores = [r.score for r in current_results]

    # Ensure we have enough data
    if len(baseline_scores) < 2 or len(current_scores) < 2:
        raise ValueError(f"Need at least 2 samples in each group. Baseline: {len(baseline_scores)}, Current: {len(current_scores)}")
        
    # Calculate current statistics
    current_stats = calculate_statistics(current_scores)
    
    # Run statistical tests
    t_stat, p_value = welch_ttest(baseline_scores, current_scores)
    effect_size = cohens_d(baseline_scores, current_scores)
    
    # Determine if drift detected
    is_significant = p_value < significance_level
    
    # Determine severity
    abs_effect = abs(effect_size)
    if not is_significant:
        severity = DriftSeverity.NONE
    elif abs_effect < 0.5:
        severity = DriftSeverity.MINOR
    elif abs_effect < 0.8:
        severity = DriftSeverity.MODERATE
    else:
        severity = DriftSeverity.MAJOR
    
    drift_detected = severity != DriftSeverity.NONE
    
    # Calculate change percentage
    change_percent = (
        (current_stats.mean - baseline.overall_stats.mean) / 
        baseline.overall_stats.mean * 100
    )
    
    # Generate summary
    direction = "improved" if change_percent > 0 else "degraded"
    if drift_detected:
        summary = (
            f"Performance has {interpret_cohens_d(effect_size)} "
            f"({abs(change_percent):.1f}% {direction}). "
            f"Change is {interpret_pvalue(p_value)}."
        )
    else:
        summary = (
            f"No significant drift detected. "
            f"Performance is stable at {current_stats.mean:.1%} "
            f"({interpret_pvalue(p_value)})."
        )
    
    # Test period description
    test_period = f"{current_runs_data[0].timestamp[:10]} to {current_runs_data[-1].timestamp[:10]}"
    
    return DriftResult(
        model_name=model_name,
        test_period=test_period,
        drift_detected=drift_detected,
        severity=severity,
        baseline_mean=baseline.overall_stats.mean,
        current_mean=current_stats.mean,
        change_percent=change_percent,
        p_value=p_value,
        cohens_d=effect_size,
        summary=summary,
    )


def compare_periods(
    storage: ResultsStorage,
    model_name: str,
    period1_start: str,
    period1_end: str,
    period2_start: str,
    period2_end: str
) -> DriftResult:
    """
    Compare performance between two specific time periods.
    
    More flexible than detect_drift - you specify exact date ranges.
    
    Args:
        storage: ResultsStorage instance
        model_name: Which model to analyze
        period1_start: Start date for period 1 (YYYY-MM-DD)
        period1_end: End date for period 1
        period2_start: Start date for period 2
        period2_end: End date for period 2
    
    Returns:
        DriftResult comparing the two periods
    
    Example:
        result = compare_periods(
            storage,
            "Claude Sonnet 4",
            "2024-11-24", "2024-11-30",  # Week 1
            "2024-12-01", "2024-12-07"   # Week 2
        )
    """
    # Load runs for each period
    all_runs = storage.load_all_runs()
    
    period1_runs = [
        r for r in all_runs
        if period1_start <= r.timestamp[:10] <= period1_end
    ]
    period2_runs = [
        r for r in all_runs
        if period2_start <= r.timestamp[:10] <= period2_end
    ]
    
    if not period1_runs or not period2_runs:
        raise ValueError("No runs found in one or both periods")
    
    # Extract scores
    period1_scores = []
    for run in period1_runs:
        results = [r for r in run.results if r.model_name == model_name]
        period1_scores.extend([r.score for r in results])
    
    period2_scores = []
    for run in period2_runs:
        results = [r for r in run.results if r.model_name == model_name]
        period2_scores.extend([r.score for r in results])
    
    if not period1_scores or not period2_scores:
        raise ValueError(f"No results found for {model_name}")
    
    # Calculate statistics
    period1_stats = calculate_statistics(period1_scores)
    period2_stats = calculate_statistics(period2_scores)
    
    # Statistical tests
    t_stat, p_value = welch_ttest(period1_scores, period2_scores)
    effect_size = cohens_d(period1_scores, period2_scores)
    
    # Determine drift
    is_significant = p_value < 0.05
    abs_effect = abs(effect_size)
    
    if not is_significant:
        severity = DriftSeverity.NONE
    elif abs_effect < 0.5:
        severity = DriftSeverity.MINOR
    elif abs_effect < 0.8:
        severity = DriftSeverity.MODERATE
    else:
        severity = DriftSeverity.MAJOR
    
    drift_detected = severity != DriftSeverity.NONE
    
    # Change percentage
    change_percent = (
        (period2_stats.mean - period1_stats.mean) / 
        period1_stats.mean * 100
    )
    
    # Summary
    direction = "improved" if change_percent > 0 else "degraded"
    if drift_detected:
        summary = (
            f"Performance has {interpret_cohens_d(effect_size)} between periods "
            f"({abs(change_percent):.1f}% {direction}). "
            f"Change is {interpret_pvalue(p_value)}."
        )
    else:
        summary = f"No significant change between periods. {interpret_pvalue(p_value)}."
    
    return DriftResult(
        model_name=model_name,
        test_period=f"{period1_start}/{period1_end} vs {period2_start}/{period2_end}",
        drift_detected=drift_detected,
        severity=severity,
        baseline_mean=period1_stats.mean,
        current_mean=period2_stats.mean,
        change_percent=change_percent,
        p_value=p_value,
        cohens_d=effect_size,
        summary=summary,
    )


def print_drift_report(result: DriftResult):
    """
    Print a human-readable drift detection report.
    
    Args:
        result: DriftResult to display
    """
    print(f"\n{'='*60}")
    print(f"DRIFT DETECTION REPORT: {result.model_name}")
    print(f"{'='*60}")
    print(f"Period: {result.test_period}")
    print()
    
    # Status
    if result.drift_detected:
        icon = "🚨"
        status = f"DRIFT DETECTED ({result.severity.value.upper()})"
    else:
        icon = "✅"
        status = "NO DRIFT DETECTED"
    
    print(f"{icon} {status}")
    print()
    
    # Performance change
    print("Performance Change:")
    print(f"  Baseline: {result.baseline_mean:.1%}")
    print(f"  Current:  {result.current_mean:.1%}")
    print(f"  Change:   {result.change_percent:+.1f}%")
    print()
    
    # Statistical evidence
    print("Statistical Evidence:")
    print(f"  p-value:   {result.p_value:.4f} ({interpret_pvalue(result.p_value)})")
    print(f"  Cohen's d: {result.cohens_d:.3f} ({interpret_cohens_d(result.cohens_d)})")
    print()
    
    # Summary
    print("Summary:")
    print(f"  {result.summary}")
    print()
    
    # Recommendation
    if result.severity == DriftSeverity.MAJOR:
        print("⚠️  RECOMMENDATION: Investigate immediately. Major performance change.")
    elif result.severity == DriftSeverity.MODERATE:
        print("⚠️  RECOMMENDATION: Monitor closely. Notable performance change.")
    elif result.severity == DriftSeverity.MINOR:
        print("ℹ️  RECOMMENDATION: Keep monitoring. Minor but significant change.")
    else:
        print("✅ RECOMMENDATION: Continue normal monitoring.")


# Command-line tool
if __name__ == "__main__":
    import sys
    from ..storage import ResultsStorage
    
    storage = ResultsStorage()
    runs = storage.load_all_runs()
    
    if len(runs) < 10:
        print(f"⚠️  Only {len(runs)} runs available.")
        print("   Need at least 10 runs (7 baseline + 3 current) for drift detection.")
        print("   Keep running daily tests and check back later!")
        sys.exit(0)
    
    print("Running drift detection analysis...")
    
    # Get all models
    model_names = set()
    for run in runs:
        model_names.update(run.models_tested)
    
    # Detect drift for each model
    for model_name in model_names:
        try:
            result = detect_drift(storage, model_name)
            print_drift_report(result)
        except Exception as e:
            print(f"Error analyzing {model_name}: {e}")