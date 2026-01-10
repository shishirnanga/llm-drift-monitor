"""
src/runner.py

The main test runner that orchestrates everything.

This module:
1. Loads test cases
2. Initializes models
3. Runs each test against each model
4. Scores responses
5. Saves results

This is the core engine of the drift monitor.
"""

from datetime import datetime
from typing import List, Optional
from tqdm import tqdm  # Progress bar library

from .models import BaseModel, ModelResponse, get_all_models, get_model_by_name
from .tests import TestCase, get_all_tests, get_tests_by_category, score_response, TestCategory
from .storage import TestResult, TestRun, ResultsStorage, generate_run_id


class DriftMonitorRunner:
    """
    Main test runner for the LLM Drift Monitor.
    
    This class orchestrates the entire testing process:
    - Loading models and tests
    - Running tests
    - Scoring results
    - Saving data
    
    Usage:
        runner = DriftMonitorRunner()
        run = runner.run_all_tests()
        print(f"Completed {len(run.results)} tests")
    """
    
    def __init__(
        self,
        models: Optional[List[BaseModel]] = None,
        data_dir: str = "./data",
        verbose: bool = True
    ):
        """
        Initialize the runner.
        
        Args:
            models: List of models to test. If None, uses all available models.
            data_dir: Directory for storing results
            verbose: Whether to print progress
        """
        self.models = models or get_all_models()
        self.storage = ResultsStorage(data_dir)
        self.verbose = verbose
        
        if not self.models:
            raise ValueError("No models available. Check your API keys.")
        
        if self.verbose:
            print(f"Initialized with {len(self.models)} model(s):")
            for model in self.models:
                print(f"  - {model.name}")
    
    def run_single_test(self, test: TestCase, model: BaseModel) -> TestResult:
        """
        Run a single test against a single model.
        
        Args:
            test: The test case to run
            model: The model to test
        
        Returns:
            TestResult with all data
        """
        # Query the model
        response: ModelResponse = model.query(test.prompt)
        
        # Score the response
        score = score_response(test, response.response)
        
        # Create result object
        return TestResult(
            test_id=test.id,
            model_name=model.name,
            model_id=model.model_id,
            timestamp=response.timestamp,
            prompt=test.prompt,
            response=response.response,
            expected=test.expected,
            score=score,
            latency_ms=response.latency_ms,
            tokens_input=response.tokens_input,
            tokens_output=response.tokens_output,
            tokens_total=response.tokens_total,
            success=response.success,
            error=response.error,
            category=test.category.value if hasattr(test.category, 'value') else test.category,
        )
    
    def run_all_tests(
        self,
        tests: Optional[List[TestCase]] = None,
        save: bool = True
    ) -> TestRun:
        """
        Run all tests against all models.
        
        Args:
            tests: Specific tests to run. If None, runs all tests.
            save: Whether to save results to disk
        
        Returns:
            TestRun containing all results
        """
        tests = tests or get_all_tests()
        run_id = generate_run_id()
        timestamp = datetime.now().isoformat()
        results: List[TestResult] = []
        
        total_iterations = len(tests) * len(self.models)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"LLM DRIFT MONITOR - Test Run")
            print(f"{'='*60}")
            print(f"Run ID: {run_id}")
            print(f"Tests: {len(tests)}")
            print(f"Models: {len(self.models)}")
            print(f"Total iterations: {total_iterations}")
            print(f"{'='*60}\n")
        
        # Create progress bar
        if self.verbose:
            pbar = tqdm(total=total_iterations, desc="Running tests")
        
        for model in self.models:
            if self.verbose:
                tqdm.write(f"\n Testing: {model.name}")
            
            for test in tests:
                result = self.run_single_test(test, model)
                results.append(result)
                
                # Update progress bar with result indicator
                if self.verbose:
                    status = "" if result.score == 1.0 else "" if result.score > 0 else ""
                    pbar.set_postfix({
                        "test": test.id[:15],
                        "score": f"{result.score:.0%}",
                        "status": status
                    })
                    pbar.update(1)
        
        if self.verbose:
            pbar.close()
        
        # Create TestRun object
        run = TestRun(
            run_id=run_id,
            timestamp=timestamp,
            models_tested=[m.name for m in self.models],
            total_tests=len(tests),
            total_results=len(results),
            results=results,
            metadata={
                "models": [{"name": m.name, "id": m.model_id} for m in self.models],
            }
        )
        
        # Save results
        if save:
            filepath = self.storage.save_run(run)
            if self.verbose:
                print(f"\n💾 Results saved to: {filepath}")
        
        # Print summary
        if self.verbose:
            self._print_summary(run)
        
        return run
    
    def run_category(self, category: TestCategory, save: bool = True) -> TestRun:
        """
        Run tests for a specific category only.
        
        Args:
            category: The TestCategory to run
            save: Whether to save results
        
        Returns:
            TestRun for that category
        """
        tests = get_tests_by_category(category)
        return self.run_all_tests(tests=tests, save=save)
    
    def _print_summary(self, run: TestRun):
        """Print a summary of the test run."""
        summary = run.calculate_summary()
        
        print(f"\n{'='*60}")
        print(" RESULTS SUMMARY")
        print(f"{'='*60}")
        
        print(f"\nTotal results: {summary['total_results']}")
        
        print("\n📈 By Model:")
        for model, stats in summary["by_model"].items():
            print(f"  {model}:")
            print(f"    Average score: {stats['avg_score']:.1%}")
            print(f"    Perfect scores: {stats['perfect_scores']}/{stats['count']}")
            print(f"    Avg latency: {stats['avg_latency_ms']:.0f}ms")
        
        print("\n📂 By Category:")
        for cat, stats in summary["by_category"].items():
            print(f"  {cat}: {stats['avg_score']:.1%} ({stats['count']} tests)")


def run_quick_test(model_name: str = "claude") -> TestRun:
    """
    Run a quick test with just a few test cases.
    
    Useful for verifying setup works before running the full suite.
    
    Args:
        model_name: Which model to test ("gpt4" or "claude")
    
    Returns:
        TestRun with results
    """
    model = get_model_by_name(model_name)
    runner = DriftMonitorRunner(models=[model])
    
    # Just run 5 tests
    tests = get_all_tests()[:5]
    return runner.run_all_tests(tests=tests, save=False)


# Main entry point when running this file directly
if __name__ == "__main__":
    print("Running quick test to verify setup...")
    
    try:
        run = run_quick_test("claude")
        print("\n Quick test completed successfully!")
        print(f"   Ran {len(run.results)} tests")
        avg_score = sum(r.score for r in run.results) / len(run.results)
        print(f"   Average score: {avg_score:.1%}")
    except Exception as e:
        print(f"\n Quick test failed: {e}")
        raise