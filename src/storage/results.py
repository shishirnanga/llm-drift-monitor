"""
src/storage/results.py

Handle saving and loading test results.

STORAGE DESIGN:
- Each test run is saved as a separate JSON file
- Files are named by timestamp: run_20241106_143022.json
- This makes it easy to track results over time
- JSON is human-readable and easy to process

DIRECTORY STRUCTURE:
data/
├── raw/           # Individual run files (one per test run)
│   ├── run_20241106_143022.json
│   ├── run_20241107_090000.json
│   └── ...
└── processed/     # Aggregated/analyzed data
    ├── daily_summary.json
    └── ...
"""

import os
import json
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class TestResult:
    """
    Result of running a single test against a single model.
    
    This captures everything we need to know about one test execution:
    - What test was run
    - Which model was tested
    - What the model said
    - How it scored
    - Performance metrics (latency, tokens)
    """
    test_id: str
    model_name: str
    model_id: str
    timestamp: str
    prompt: str
    response: str
    expected: Optional[str]
    score: float
    latency_ms: int
    tokens_input: int
    tokens_output: int
    tokens_total: int
    success: bool
    error: Optional[str] = None
    category: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestResult":
        """Create TestResult from dictionary."""
        return cls(**data)


@dataclass 
class TestRun:
    """
    A complete test run (all tests against all models at one point in time).
    
    Contains metadata about the run plus all individual results.
    """
    run_id: str
    timestamp: str
    models_tested: List[str]
    total_tests: int
    total_results: int
    results: List[TestResult]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "models_tested": self.models_tested,
            "total_tests": self.total_tests,
            "total_results": self.total_results,
            "results": [r.to_dict() for r in self.results],
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestRun":
        """Create TestRun from dictionary."""
        results = [TestResult.from_dict(r) for r in data.get("results", [])]
        return cls(
            run_id=data["run_id"],
            timestamp=data["timestamp"],
            models_tested=data.get("models_tested", []),
            total_tests=data.get("total_tests", 0),
            total_results=data.get("total_results", len(results)),
            results=results,
            metadata=data.get("metadata", {}),
        )
    
    def get_results_by_model(self, model_name: str) -> List[TestResult]:
        """Get all results for a specific model."""
        return [r for r in self.results if r.model_name == model_name]
    
    def get_results_by_category(self, category: str) -> List[TestResult]:
        """Get all results for a specific category."""
        return [r for r in self.results if r.category == category]
    
    def calculate_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics for this run."""
        summary = {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "total_results": len(self.results),
            "by_model": {},
            "by_category": {},
        }
        
        # By model
        for model in self.models_tested:
            model_results = self.get_results_by_model(model)
            if model_results:
                scores = [r.score for r in model_results]
                latencies = [r.latency_ms for r in model_results]
                summary["by_model"][model] = {
                    "count": len(model_results),
                    "avg_score": sum(scores) / len(scores),
                    "avg_latency_ms": sum(latencies) / len(latencies),
                    "perfect_scores": sum(1 for s in scores if s == 1.0),
                }
        
        # By category
        categories = set(r.category for r in self.results if r.category)
        for cat in categories:
            cat_results = self.get_results_by_category(cat)
            if cat_results:
                scores = [r.score for r in cat_results]
                summary["by_category"][cat] = {
                    "count": len(cat_results),
                    "avg_score": sum(scores) / len(scores),
                }
        
        return summary


class ResultsStorage:
    """
    Handles saving and loading test results.
    
    Usage:
        storage = ResultsStorage("./data")
        
        # Save a run
        storage.save_run(test_run)
        
        all_runs = storage.load_all_runs()
        
        recent_runs = storage.load_runs_since("2024-11-01")
    """
    
    def __init__(self, data_dir: str = "./data"):
        """
        Initialize storage.
        
        Args:
            data_dir: Base directory for data storage
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def save_run(self, run: TestRun) -> str:
        """
        Save a test run to disk.
        
        Args:
            run: The TestRun to save
        
        Returns:
            Path to the saved file
        """
        filename = f"run_{run.run_id}.json"
        filepath = self.raw_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(run.to_dict(), f, indent=2)
        
        return str(filepath)
    
    def load_run(self, run_id: str) -> Optional[TestRun]:
        """
        Load a specific test run by ID.
        
        Args:
            run_id: The run ID (timestamp format)
        
        Returns:
            TestRun if found, None otherwise
        """
        filename = f"run_{run_id}.json"
        filepath = self.raw_dir / filename
        
        if not filepath.exists():
            return None
        
        with open(filepath) as f:
            data = json.load(f)
        
        return TestRun.from_dict(data)
    
    def load_all_runs(self) -> List[TestRun]:
        """
        Load all test runs from disk.
        
        Returns:
            List of TestRun objects, sorted by timestamp (oldest first)
        """
        runs = []
        
        for filepath in sorted(self.raw_dir.glob("run_*.json")):
            try:
                with open(filepath) as f:
                    data = json.load(f)
                runs.append(TestRun.from_dict(data))
            except Exception as e:
                print(f"Warning: Could not load {filepath}: {e}")
        
        return runs
    
    def load_runs_since(self, since_date: str) -> List[TestRun]:
        """
        Load runs since a specific date.
        
        Args:
            since_date: Date string in format "YYYY-MM-DD"
        
        Returns:
            List of TestRun objects from that date onward
        """
        all_runs = self.load_all_runs()
        
        return [
            run for run in all_runs
            if run.timestamp[:10] >= since_date
        ]
    
    def get_all_results_flat(self) -> List[Dict[str, Any]]:
        """
        Get all results as a flat list of dictionaries.
        
        Useful for loading into pandas DataFrame.
        
        Returns:
            List of result dictionaries with run_id added
        """
        all_results = []
        
        for run in self.load_all_runs():
            for result in run.results:
                result_dict = result.to_dict()
                result_dict["run_id"] = run.run_id
                all_results.append(result_dict)
        
        return all_results
    
    def get_latest_run(self) -> Optional[TestRun]:
        """Get the most recent test run."""
        runs = self.load_all_runs()
        return runs[-1] if runs else None
    
    def count_runs(self) -> int:
        """Count total number of saved runs."""
        return len(list(self.raw_dir.glob("run_*.json")))


def generate_run_id() -> str:
    """
    Generate a unique run ID based on current timestamp.
    
    Format: YYYYMMDD_HHMMSS
    Example: 20241106_143022
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# Quick test
if __name__ == "__main__":
    # Create a sample result
    result = TestResult(
        test_id="math_001",
        model_name="GPT-4 Turbo",
        model_id="gpt-4-turbo-preview",
        timestamp=datetime.now().isoformat(),
        prompt="What is 2+2?",
        response="4",
        expected="4",
        score=1.0,
        latency_ms=500,
        tokens_input=10,
        tokens_output=5,
        tokens_total=15,
        success=True,
        category="math",
    )
    
    # Create a sample run
    run = TestRun(
        run_id=generate_run_id(),
        timestamp=datetime.now().isoformat(),
        models_tested=["GPT-4 Turbo"],
        total_tests=1,
        total_results=1,
        results=[result],
    )
    
    # Test storage
    storage = ResultsStorage("./data")
    filepath = storage.save_run(run)
    print(f" Saved to: {filepath}")
    
    # Test loading
    loaded = storage.load_all_runs()
    print(f" Loaded {len(loaded)} runs")