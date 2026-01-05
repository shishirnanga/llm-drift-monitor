"""
src/analysis/statistics.py

Core statistical functions for drift detection.

This module implements the statistical tests we need:
- Welch's t-test (doesn't assume equal variance)
- Cohen's d (effect size)
- Confidence intervals
- Basic descriptive statistics
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Statistics:
    """Container for statistical measurements."""
    mean: float
    std: float
    variance: float
    min: float
    max: float
    count: int
    confidence_interval_95: Tuple[float, float]
    
    def __repr__(self) -> str:
        return (
            f"Statistics(mean={self.mean:.3f}, std={self.std:.3f}, "
            f"n={self.count}, CI95={self.confidence_interval_95})"
        )


def calculate_statistics(values: List[float]) -> Statistics:
    """
    Calculate descriptive statistics for a list of values.
    
    Args:
        values: List of numeric values (e.g., scores)
    
    Returns:
        Statistics object with all measurements
    """
    if not values:
        raise ValueError("Cannot calculate statistics on empty list")
    
    arr = np.array(values)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))  # ddof=1 for sample std
    
    # Calculate 95% confidence interval
    # CI = mean ± (critical_value * standard_error)
    # For 95% CI, critical value ≈ 1.96 for large n
    # For small n, use t-distribution
    n = len(arr)
    if n > 30:
        # Use normal distribution (z-score)
        critical_value = 1.96
    else:
        # Use t-distribution
        critical_value = stats.t.ppf(0.975, df=n-1)  # 0.975 for two-tailed 95%
    
    standard_error = std / np.sqrt(n)
    margin_of_error = critical_value * standard_error
    ci_lower = mean - margin_of_error
    ci_upper = mean + margin_of_error
    
    return Statistics(
        mean=mean,
        std=std,
        variance=float(np.var(arr, ddof=1)),
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        count=n,
        confidence_interval_95=(ci_lower, ci_upper),
    )


def welch_ttest(group1: List[float], group2: List[float]) -> Tuple[float, float]:
    """
    Perform Welch's t-test (doesn't assume equal variance).
    
    Tests whether two groups have significantly different means.
    
    Args:
        group1: First group of values (e.g., baseline scores)
        group2: Second group of values (e.g., current scores)
    
    Returns:
        Tuple of (t_statistic, p_value)
        
        p_value < 0.05 means statistically significant difference
        p_value < 0.01 means highly significant
        p_value < 0.001 means very highly significant
    
    Example:
        baseline = [0.9, 0.85, 0.88, 0.92, 0.87]
        current = [0.75, 0.78, 0.73, 0.76, 0.74]
        t_stat, p_value = welch_ttest(baseline, current)
        
        if p_value < 0.05:
            print("Performance has significantly changed!")
    """
    if len(group1) < 2 or len(group2) < 2:
        raise ValueError("Need at least 2 samples in each group")
    
    # Welch's t-test (unequal variances)
    t_statistic, p_value = stats.ttest_ind(group1, group2, equal_var=False)
    
    return float(t_statistic), float(p_value)


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """
    Calculate Cohen's d (standardized effect size).
    
    Measures HOW MUCH the groups differ, not just whether they differ.
    
    Interpretation:
        |d| < 0.2: negligible
        |d| < 0.5: small effect
        |d| < 0.8: medium effect
        |d| >= 0.8: large effect
    
    Args:
        group1: First group (e.g., baseline)
        group2: Second group (e.g., current)
    
    Returns:
        Cohen's d value
        
        Positive = group2 higher than group1
        Negative = group2 lower than group1
    
    Example:
        baseline = [0.9, 0.85, 0.88, 0.92]
        current = [0.75, 0.78, 0.73, 0.76]
        d = cohens_d(baseline, current)
        # d ≈ -1.5 (large negative effect = performance dropped)
    """
    arr1 = np.array(group1)
    arr2 = np.array(group2)
    
    mean1 = np.mean(arr1)
    mean2 = np.mean(arr2)
    
    # Pooled standard deviation
    n1, n2 = len(arr1), len(arr2)
    var1 = np.var(arr1, ddof=1)
    var2 = np.var(arr2, ddof=1)
    
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Cohen's d
    d = (mean2 - mean1) / pooled_std
    
    return float(d)


def interpret_cohens_d(d: float) -> str:
    """
    Interpret Cohen's d value in plain English.
    
    Args:
        d: Cohen's d value
    
    Returns:
        Human-readable interpretation
    """
    abs_d = abs(d)
    direction = "increased" if d > 0 else "decreased"
    
    if abs_d < 0.2:
        magnitude = "negligibly"
    elif abs_d < 0.5:
        magnitude = "slightly"
    elif abs_d < 0.8:
        magnitude = "moderately"
    else:
        magnitude = "substantially"
    
    return f"{magnitude} {direction}"


def interpret_pvalue(p: float) -> str:
    """
    Interpret p-value in plain English.
    
    Args:
        p: p-value from statistical test
    
    Returns:
        Human-readable interpretation
    """
    if p < 0.001:
        return "very highly significant (p < 0.001)"
    elif p < 0.01:
        return "highly significant (p < 0.01)"
    elif p < 0.05:
        return "statistically significant (p < 0.05)"
    elif p < 0.10:
        return "marginally significant (p < 0.10)"
    else:
        return "not statistically significant (p ≥ 0.10)"


# Quick test
if __name__ == "__main__":
    print("Testing statistical functions...\n")
    
    # Simulate baseline vs degraded performance
    baseline = [0.90, 0.92, 0.88, 0.91, 0.89, 0.93, 0.87]
    current = [0.78, 0.82, 0.75, 0.80, 0.76, 0.81, 0.77]
    
    print("Baseline scores:", baseline)
    print("Current scores:", current)
    print()
    
    # Calculate statistics
    baseline_stats = calculate_statistics(baseline)
    current_stats = calculate_statistics(current)
    
    print(f"Baseline: {baseline_stats}")
    print(f"Current:  {current_stats}")
    print()
    
    # Run t-test
    t_stat, p_value = welch_ttest(baseline, current)
    print(f"Welch's t-test:")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Interpretation: {interpret_pvalue(p_value)}")
    print()
    
    # Calculate effect size
    d = cohens_d(baseline, current)
    print(f"Cohen's d: {d:.3f}")
    print(f"  Interpretation: Performance {interpret_cohens_d(d)}")
    print()
    
    if p_value < 0.05 and abs(d) > 0.5:
        print("🚨 DRIFT DETECTED: Significant and meaningful change in performance")
    elif p_value < 0.05:
        print("⚠️  Statistically significant but small effect size")
    else:
        print("✅ No significant drift detected")