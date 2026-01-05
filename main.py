#!/usr/bin/env python3
"""
main.py

LLM Drift Monitor - Main Entry Point

This is the script you run to execute the test suite.
It can be run directly: python main.py

Options:
  --quick     Run a quick test (5 tests) to verify setup
  --model     Specify model(s) to test: gpt4, claude, or all
  --category  Test specific category: math, reasoning, factual, consistency, instruction
  --no-save   Don't save results to disk

Examples:
  python main.py                    # Run full suite against all models
  python main.py --quick            # Quick verification test
  python main.py --model claude     # Test only Claude
  python main.py --category math    # Run only math tests
"""

import argparse
import sys
from datetime import datetime

# Add src to path so imports work
sys.path.insert(0, '.')

from src.models import OpenAIModel, AnthropicModel, get_all_models
from src.tests import get_all_tests, get_tests_by_category, TestCategory
from src.runner import DriftMonitorRunner, run_quick_test


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LLM Drift Monitor - Track how AI models change over time",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    Run full test suite
  python main.py --quick            Quick verification (5 tests)
  python main.py --model claude     Test only Claude
  python main.py --category math    Run only math tests
        """
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick test (5 tests) to verify setup"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        choices=["gpt4", "claude", "all"],
        default="all",
        help="Which model(s) to test (default: all)"
    )
    
    parser.add_argument(
        "--category",
        type=str,
        choices=["math", "reasoning", "factual", "consistency", "instruction"],
        default=None,
        help="Test only a specific category"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to disk"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output"
    )
    
    return parser.parse_args()


def get_models_from_arg(model_arg: str):
    """Get model instances based on command line argument."""
    if model_arg == "all":
        return get_all_models()
    elif model_arg == "gpt4":
        return [OpenAIModel()]
    elif model_arg == "claude":
        return [AnthropicModel()]
    else:
        raise ValueError(f"Unknown model: {model_arg}")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Print header
    if not args.quiet:
        print("""
╔═══════════════════════════════════════════════════════════════╗
║                    LLM DRIFT MONITOR                          ║
║          Track how AI models change over time                 ║
╚═══════════════════════════════════════════════════════════════╝
        """)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Quick test mode
    if args.quick:
        if not args.quiet:
            print("\n🚀 Running quick test (5 tests per model)...\n")
        
        models = get_all_models()
        
        for model in models:
            run = run_quick_test(model.name)
            
            if not args.quiet:
                avg = sum(r.score for r in run.results) / len(run.results) if run.results else 0
                print(f"✅ {model.name}: {avg:.1%} avg score")
        
        if not args.quiet:
            print("\n💡 Run without --quick for full test suite on all models")
        
        return 0
    
    # Full test mode
    try:
        # Get models
        models = get_models_from_arg(args.model)
        
        if not models:
            print("❌ No models available. Check your API keys in .env")
            return 1
        
        # Get tests
        if args.category:
            category = TestCategory(args.category)
            tests = get_tests_by_category(category)
            if not args.quiet:
                print(f"\n📂 Running {args.category} tests only ({len(tests)} tests)")
        else:
            tests = get_all_tests()
        
        # Create runner
        runner = DriftMonitorRunner(
            models=models,
            verbose=not args.quiet
        )
        
        # Run tests
        run = runner.run_all_tests(
            tests=tests,
            save=not args.no_save
        )
        
        # Final summary
        if not args.quiet:
            print(f"\n{'='*60}")
            print("✅ TEST RUN COMPLETE")
            print(f"{'='*60}")
            print(f"Run ID: {run.run_id}")
            print(f"Total results: {len(run.results)}")
            
            # Overall score
            avg_score = sum(r.score for r in run.results) / len(run.results)
            print(f"Overall average score: {avg_score:.1%}")
            
            if not args.no_save:
                print(f"\n📁 Results saved to: data/raw/run_{run.run_id}.json")
            
            print("\n💡 Next steps:")
            print("   - Run 'python dashboard/app.py' to view results")
            print("   - Set up daily runs with: scripts/schedule.sh")
            print("   - Run again tomorrow to start tracking drift!")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())