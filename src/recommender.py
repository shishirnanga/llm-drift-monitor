"""
Model Recommendation Engine

Analyzes performance data and recommends the best model for specific tasks.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
from src.storage import ResultsStorage


class TaskType(Enum):
    """Categories of tasks for recommendations."""
    MATH = "math"
    REASONING = "reasoning"
    FACTUAL = "factual"
    CREATIVE_WRITING = "creative"
    CODING = "code"
    INSTRUCTION_FOLLOWING = "instruction"
    CONSISTENCY = "consistency"


@dataclass
class ModelRecommendation:
    """Recommendation result."""
    recommended_model: str
    score: float
    confidence: float
    reasoning: str
    alternatives: List[tuple[str, float]]  # (model_name, score)
    performance_details: Dict[str, float]


class ModelRecommender:
    """Recommends best model for specific tasks based on performance data."""
    
    def __init__(self, storage: ResultsStorage):
        self.storage = storage
        self.performance_matrix = self._calculate_performance_matrix()
    
    def _calculate_performance_matrix(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Calculate performance matrix: model -> category -> metrics
        
        Returns:
            {
                "GPT-4 Turbo": {
                    "math": {"accuracy": 0.92, "latency": 543, "consistency": 0.95},
                    ...
                },
                ...
            }
        """
        runs = self.storage.load_all_runs()
        
        if not runs:
            return {}
        
        # Aggregate results by model and category
        model_category_results = {}
        
        for run in runs:
            for result in run.results:
                model = result.model_name
                category = result.test_id.split('_')[0]  # e.g., "math" from "math_001"
                
                if model not in model_category_results:
                    model_category_results[model] = {}
                
                if category not in model_category_results[model]:
                    model_category_results[model][category] = {
                        'scores': [],
                        'latencies': [],
                    }
                
                model_category_results[model][category]['scores'].append(result.score)
                model_category_results[model][category]['latencies'].append(result.latency_ms)
        
        performance_matrix = {}
        
        for model, categories in model_category_results.items():
            performance_matrix[model] = {}
            
            for category, data in categories.items():
                scores = data['scores']
                latencies = data['latencies']
                
                avg_accuracy = sum(scores) / len(scores) if scores else 0
                avg_latency = sum(latencies) / len(latencies) if latencies else 0
                
                # Consistency = 1 - (std_dev / mean)
                if len(scores) > 1:
                    mean = avg_accuracy
                    variance = sum((x - mean) ** 2 for x in scores) / len(scores)
                    std_dev = variance ** 0.5
                    consistency = 1 - (std_dev / mean if mean > 0 else 0)
                else:
                    consistency = 1.0
                
                performance_matrix[model][category] = {
                    'accuracy': avg_accuracy,
                    'latency': avg_latency,
                    'consistency': max(0, min(1, consistency))
                }
        
        return performance_matrix
    
    def recommend(
        self,
        task_type: TaskType,
        priorities: Optional[Dict[str, float]] = None
    ) -> ModelRecommendation:
        """
        Recommend the best model for a specific task type.
        
        Args:
            task_type: Type of task
            priorities: Weight for each metric (accuracy, speed, consistency)
                       Defaults to {"accuracy": 0.7, "speed": 0.2, "consistency": 0.1}
        
        Returns:
            ModelRecommendation object
        """
        if priorities is None:
            priorities = {
                "accuracy": 0.7,
                "speed": 0.2,
                "consistency": 0.1
            }
        
        category = task_type.value
        
        model_scores = {}
        
        for model, categories in self.performance_matrix.items():
            if category not in categories:
                continue
            
            metrics = categories[category]
            
            # Normalize latency (lower is better, convert to 0-1 scale)
            # Find max latency across all models for this category
            all_latencies = [
                self.performance_matrix[m][category]['latency']
                for m in self.performance_matrix
                if category in self.performance_matrix[m]
            ]
            max_latency = max(all_latencies) if all_latencies else 1
            
            # Speed score (inverted latency)
            speed_score = 1 - (metrics['latency'] / max_latency) if max_latency > 0 else 1
            
            weighted_score = (
                metrics['accuracy'] * priorities.get("accuracy", 0.7) +
                speed_score * priorities.get("speed", 0.2) +
                metrics['consistency'] * priorities.get("consistency", 0.1)
            )
            
            model_scores[model] = {
                'total_score': weighted_score,
                'accuracy': metrics['accuracy'],
                'speed': speed_score,
                'consistency': metrics['consistency'],
                'latency_ms': metrics['latency']
            }
        
        if not model_scores:
            raise ValueError(f"No data available for task type: {task_type.value}")
        
        # Find best model
        best_model = max(model_scores.items(), key=lambda x: x[1]['total_score'])
        best_model_name = best_model[0]
        best_score = best_model[1]['total_score']
        
        alternatives = sorted(
            [(m, s['total_score']) for m, s in model_scores.items() if m != best_model_name],
            key=lambda x: x[1],
            reverse=True
        )
        
        if alternatives:
            second_best_score = alternatives[0][1]
            confidence = min(1.0, (best_score - second_best_score) / best_score) if best_score > 0 else 0
        else:
            confidence = 1.0
        
        # Generate reasoning
        metrics = model_scores[best_model_name]
        reasoning = self._generate_reasoning(
            best_model_name,
            task_type,
            metrics,
            confidence
        )
        
        return ModelRecommendation(
            recommended_model=best_model_name,
            score=best_score,
            confidence=confidence,
            reasoning=reasoning,
            alternatives=alternatives[:2],  # Top 2 alternatives
            performance_details=metrics
        )
    
    def _generate_reasoning(
        self,
        model_name: str,
        task_type: TaskType,
        metrics: Dict[str, float],
        confidence: float
    ) -> str:
        """Generate human-readable reasoning for the recommendation."""
        
        strengths = []
        if metrics['accuracy'] > 0.9:
            strengths.append(f"highest accuracy ({metrics['accuracy']:.1%})")
        if metrics['speed'] > 0.8:
            strengths.append(f"fast response time ({metrics['latency_ms']:.0f}ms)")
        if metrics['consistency'] > 0.9:
            strengths.append(f"very consistent performance")
        
        strength_text = ", ".join(strengths) if strengths else "balanced performance"
        
        # Confidence level description
        if confidence > 0.8:
            conf_text = "clear best choice"
        elif confidence > 0.5:
            conf_text = "recommended choice"
        else:
            conf_text = "slight edge over alternatives"
        
        return (
            f"{model_name} excels at this task with {strength_text}. "
            f"This is the {conf_text} for {task_type.value} tasks."
        )
    
    def get_all_recommendations(self) -> Dict[TaskType, ModelRecommendation]:
        """Get recommendations for all task types."""
        recommendations = {}
        
        for task_type in TaskType:
            try:
                recommendations[task_type] = self.recommend(task_type)
            except ValueError:
                continue  # Skip if no data for this task type
        
        return recommendations
    
    def print_recommendations(self):
        """Print all recommendations in a formatted way."""
        print("\n" + "=" * 60)
        print(" AI MODEL RECOMMENDER")
        print("=" * 60)
        
        all_recs = self.get_all_recommendations()
        
        for task_type, rec in all_recs.items():
            print(f"\n {task_type.value.upper()}")
            print(f"   Best: {rec.recommended_model}")
            print(f"   Performance: {rec.performance_details['accuracy']:.1%}")
            print(f"   Confidence: {rec.confidence:.1%}")
            print(f"   Latency: {rec.performance_details['latency_ms']:.0f}ms")
            print(f"   Reason: {rec.reasoning}")
            
            if rec.alternatives:
                print(f"   Alternatives:")
                for alt_model, alt_score in rec.alternatives:
                    print(f"     - {alt_model} ({alt_score:.1%}): Consider if you prioritize different metrics")


# Example usage
if __name__ == "__main__":
    from src.storage import ResultsStorage
    
    storage = ResultsStorage()
    recommender = ModelRecommender(storage)
    
    # Print all recommendations
    recommender.print_recommendations()
    
    print("\n" + "=" * 60)
    print("CUSTOM RECOMMENDATION EXAMPLE")
    print("=" * 60)
    
    rec = recommender.recommend(
        TaskType.MATH,
        priorities={"accuracy": 0.9, "speed": 0.05, "consistency": 0.05}
    )
    
    print(f"\nFor MATH tasks (prioritizing accuracy):")
    print(f"   Recommended: {rec.recommended_model}")
    print(f"   Score: {rec.score:.3f}")
    print(f"   Confidence: {rec.confidence:.1%}")
    print(f"   {rec.reasoning}")