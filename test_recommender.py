"""
Test the recommender with your existing data
"""

import sys
sys.path.append('.')

from src.storage import ResultsStorage
from src.recommender import ModelRecommender, TaskType

def test_recommender():
    """Test the recommender system."""
    
    print("=" * 60)
    print("TESTING MODEL RECOMMENDER")
    print("=" * 60)
    
    # Load storage
    storage = ResultsStorage()
    runs = storage.load_all_runs()
    
    print(f"\n Loaded {len(runs)} test runs")
    
    if len(runs) < 3:
        print("  Warning: You need at least 3 runs for meaningful recommendations")
        print(f"   You currently have {len(runs)} runs")
        print("   Run 'python main.py' a few more times to collect data")
        return
    
    # Create recommender
    recommender = ModelRecommender(storage)
    
    # Print all recommendations
    recommender.print_recommendations()
    
    # Test custom priorities
    print("\n" + "=" * 60)
    print("CUSTOM PRIORITY EXAMPLES")
    print("=" * 60)
    
    examples = [
        ("Speed-focused", {"accuracy": 0.3, "speed": 0.6, "consistency": 0.1}),
        ("Accuracy-focused", {"accuracy": 0.9, "speed": 0.05, "consistency": 0.05}),
        ("Balanced", {"accuracy": 0.5, "speed": 0.3, "consistency": 0.2}),
    ]
    
    for name, priorities in examples:
        print(f"\n{name} (accuracy:{priorities['accuracy']:.0%}, speed:{priorities['speed']:.0%}, consistency:{priorities['consistency']:.0%}):")
        
        try:
            rec = recommender.recommend(TaskType.MATH, priorities)
            print(f"   → {rec.recommended_model} (score: {rec.score:.3f})")
        except ValueError as e:
            print(f"   → Not enough data: {e}")

if __name__ == "__main__":
    test_recommender()