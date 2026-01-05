#!/usr/bin/env python3
"""
analyze.py

Command-line tool for analyzing LLM drift data.

Usage:
    python analyze.py baseline              # Show baseline metrics
    python analyze.py drift                 # Detect drift
    python analyze.py compare 2024-11-24 2024-12-01  # Compare dates
"""

import sys
import argparse
from src.storage import ResultsStorage
from src.analysis import (
    calculate_baseline,
    get_all_baselines,
    print_baseline_report,
    detect_drift,
    compare_periods,
    print_drift_report,
)


def cmd_baseline(args):
    """Show baseline metrics."""
    storage = ResultsStorage()
    
    try:
        baselines = get_all_baselines(storage, num_runs=args.runs)
        
        for model_name, baseline in baselines.items():
            print_baseline_report(baseline)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


def cmd_drift(args):
    """Detect drift from baseline."""
    storage = ResultsStorage()
    runs = storage.load_all_runs()
    
    if len(runs) < args.baseline + args.current:
        print(f"❌ Need at least {args.baseline + args.current} runs")
        print(f"   Currently have: {len(runs)}")
        print(f"   Keep running daily and try again later!")
        return 1
    
    # Get all models
    model_names = set()
    for run in runs:
        model_names.update(run.models_tested)
    
    # Detect drift for each
    for model_name in model_names:
        try:
            result = detect_drift(
                storage,
                model_name,
                baseline_runs=args.baseline,
                current_runs=args.current
            )
            print_drift_report(result)
        except Exception as e:
            print(f"Error analyzing {model_name}: {e}")
    
    return 0


def cmd_compare(args):
    """Compare two time periods."""
    storage = ResultsStorage()
    
    # Get all models
    runs = storage.load_all_runs()
    model_names = set()
    for run in runs:
        model_names.update(run.models_tested)
    
    # Compare periods for each model
    for model_name in model_names:
        try:
            result = compare_periods(
                storage,
                model_name,
                args.period1_start,
                args.period1_end,
                args.period2_start,
                args.period2_end
            )
            print_drift_report(result)
        except Exception as e:
            print(f"Error comparing {model_name}: {e}")
    
    return 0


def cmd_summary(args):
    """Show quick summary of all data."""
    storage = ResultsStorage()
    runs = storage.load_all_runs()
    
    if not runs:
        print("No data found.")
        return 1
    
    print(f"\n{'='*60}")
    print("DATA SUMMARY")
    print(f"{'='*60}")
    print(f"Total runs: {len(runs)}")
    print(f"Date range: {runs[0].timestamp[:10]} to {runs[-1].timestamp[:10]}")
    print(f"Total data points: {sum(len(r.results) for r in runs)}")
    print()
    
    # Models
    model_names = set()
    for run in runs:
        model_names.update(run.models_tested)
    print(f"Models: {', '.join(model_names)}")
    print()
    
    # Recent performance
    if len(runs) >= 3:
        print("Recent Performance (last 3 runs):")
        for run in runs[-3:]:
            date = run.timestamp[:10]
            summary = run.calculate_summary()
            print(f"\n  {date}:")
            for model, stats in summary["by_model"].items():
                print(f"    {model}: {stats['avg_score']:.1%}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Analyze LLM drift data",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # baseline command
    baseline_parser = subparsers.add_parser('baseline', help='Show baseline metrics')
    baseline_parser.add_argument('--runs', type=int, default=7, help='Number of runs for baseline')
    
    # drift command
    drift_parser = subparsers.add_parser('drift', help='Detect drift from baseline')
    drift_parser.add_argument('--baseline', type=int, default=7, help='Baseline runs')
    drift_parser.add_argument('--current', type=int, default=3, help='Current runs to test')
    
    # compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two time periods')
    compare_parser.add_argument('period1_start', help='Period 1 start (YYYY-MM-DD)')
    compare_parser.add_argument('period1_end', help='Period 1 end (YYYY-MM-DD)')
    compare_parser.add_argument('period2_start', help='Period 2 start (YYYY-MM-DD)')
    compare_parser.add_argument('period2_end', help='Period 2 end (YYYY-MM-DD)')
    
    # summary command
    summary_parser = subparsers.add_parser('summary', help='Show data summary')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Route to appropriate command
    if args.command == 'baseline':
        return cmd_baseline(args)
    elif args.command == 'drift':
        return cmd_drift(args)
    elif args.command == 'compare':
        return cmd_compare(args)
    elif args.command == 'summary':
        return cmd_summary(args)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())