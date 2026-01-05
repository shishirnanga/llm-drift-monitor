"""
src/tests/test_cases.py

Defines all the test cases for the LLM Drift Monitor.

TEST DESIGN PHILOSOPHY:
1. Tests should be DETERMINISTIC - same input should give same output
2. Tests should be SCORABLE - we need to programmatically check correctness
3. Tests should cover DIFFERENT CAPABILITIES - math, reasoning, facts, etc.
4. Tests should be STABLE - the "right answer" shouldn't change over time

CATEGORIES:
1. MATH - Arithmetic, word problems, calculations
   Why: Easy to verify, models historically struggle with math, good drift indicator

2. REASONING - Logic puzzles, deduction, common sense
   Why: Tests deeper understanding, classic failure modes

3. FACTUAL - Knowledge recall, dates, facts
   Why: Should be stable (historical facts don't change), tests knowledge

4. CONSISTENCY - Same question asked different ways
   Why: Model should give same answer regardless of phrasing

5. INSTRUCTION - Following format constraints
   Why: Important for applications, tests instruction-following ability
"""

from dataclasses import dataclass, field
from typing import Optional, List, Callable
from enum import Enum
from .new_tests import get_all_new_tests


class TestCategory(Enum):
    """Categories of tests."""
    MATH = "math"
    REASONING = "reasoning"
    FACTUAL = "factual"
    CONSISTENCY = "consistency"
    INSTRUCTION = "instruction"


class ScoringMethod(Enum):
    """How to score a response."""
    EXACT = "exact"           # Response must contain exact answer
    NUMERIC = "numeric"       # Response must contain correct number (with tolerance)
    RUBRIC = "rubric"        # Custom scoring logic
    FORMAT = "format"         # Check format compliance


@dataclass
class TestCase:
    """
    A single test case.
    
    Attributes:
        id: Unique identifier (e.g., "math_001")
        category: Which category this test belongs to
        prompt: The question to ask the model
        expected: The expected answer (if applicable)
        scoring_method: How to evaluate the response
        description: Human-readable description of what this tests
        difficulty: Optional difficulty rating (1-5)
        tags: Optional tags for filtering
    """
    id: str
    category: TestCategory
    prompt: str
    expected: Optional[str]
    scoring_method: ScoringMethod
    description: str = ""
    difficulty: int = 1
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "category": self.category.value,
            "prompt": self.prompt,
            "expected": self.expected,
            "scoring_method": self.scoring_method.value,
            "description": self.description,
            "difficulty": self.difficulty,
            "tags": self.tags,
        }


# =============================================================================
# MATH TESTS
# =============================================================================
# These test basic arithmetic and mathematical reasoning.
# They're easy to score (numbers are either right or wrong) and 
# have historically been areas where LLMs struggle or show drift.

MATH_TESTS = [
    TestCase(
        id="math_001",
        category=TestCategory.MATH,
        prompt="What is 247 + 389? Reply with just the number.",
        expected="636",
        scoring_method=ScoringMethod.EXACT,
        description="Simple addition",
        difficulty=1,
    ),
    TestCase(
        id="math_002",
        category=TestCategory.MATH,
        prompt="What is 15% of 240? Reply with just the number.",
        expected="36",
        scoring_method=ScoringMethod.NUMERIC,
        description="Percentage calculation",
        difficulty=2,
    ),
    TestCase(
        id="math_003",
        category=TestCategory.MATH,
        prompt="If a train travels 120 miles in 2 hours, what is its average speed in miles per hour? Reply with just the number.",
        expected="60",
        scoring_method=ScoringMethod.NUMERIC,
        description="Speed calculation word problem",
        difficulty=2,
    ),
    TestCase(
        id="math_004",
        category=TestCategory.MATH,
        prompt="What is 17 × 23? Reply with just the number.",
        expected="391",
        scoring_method=ScoringMethod.EXACT,
        description="Two-digit multiplication",
        difficulty=2,
    ),
    TestCase(
        id="math_005",
        category=TestCategory.MATH,
        prompt="A shirt costs $45. It's on sale for 20% off. What is the sale price in dollars? Reply with just the number.",
        expected="36",
        scoring_method=ScoringMethod.NUMERIC,
        description="Discount calculation",
        difficulty=2,
    ),
    TestCase(
        id="math_006",
        category=TestCategory.MATH,
        prompt="What is the square root of 144? Reply with just the number.",
        expected="12",
        scoring_method=ScoringMethod.EXACT,
        description="Square root",
        difficulty=1,
    ),
    TestCase(
        id="math_007",
        category=TestCategory.MATH,
        prompt="If x + 5 = 12, what is x? Reply with just the number.",
        expected="7",
        scoring_method=ScoringMethod.EXACT,
        description="Simple algebra",
        difficulty=1,
    ),
    TestCase(
        id="math_008",
        category=TestCategory.MATH,
        prompt="What is 1000 divided by 8? Reply with just the number.",
        expected="125",
        scoring_method=ScoringMethod.EXACT,
        description="Division",
        difficulty=1,
    ),
    TestCase(
        id="math_009",
        category=TestCategory.MATH,
        prompt="What is 3 to the power of 4? Reply with just the number.",
        expected="81",
        scoring_method=ScoringMethod.EXACT,
        description="Exponentiation",
        difficulty=2,
    ),
    TestCase(
        id="math_010",
        category=TestCategory.MATH,
        prompt="A rectangle has length 8cm and width 5cm. What is its area in square centimeters? Reply with just the number.",
        expected="40",
        scoring_method=ScoringMethod.EXACT,
        description="Area calculation",
        difficulty=1,
    ),
]


# =============================================================================
# REASONING TESTS  
# =============================================================================
# These test logical reasoning and common sense.
# Many of these are classic "trick questions" that reveal how models think.

REASONING_TESTS = [
    TestCase(
        id="reason_001",
        category=TestCategory.REASONING,
        prompt="If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly? Answer only 'yes' or 'no'.",
        expected="no",
        scoring_method=ScoringMethod.RUBRIC,
        description="Syllogism - invalid conclusion",
        difficulty=3,
        tags=["logic", "syllogism"],
    ),
    TestCase(
        id="reason_002",
        category=TestCategory.REASONING,
        prompt="A is taller than B. C is shorter than B. Who is the tallest? Reply with just the letter.",
        expected="A",
        scoring_method=ScoringMethod.EXACT,
        description="Transitive comparison",
        difficulty=2,
    ),
    TestCase(
        id="reason_003",
        category=TestCategory.REASONING,
        prompt="If it takes 5 machines 5 minutes to make 5 widgets, how many minutes would it take 100 machines to make 100 widgets? Reply with just the number.",
        expected="5",
        scoring_method=ScoringMethod.NUMERIC,
        description="Classic widget problem",
        difficulty=3,
        tags=["trick question"],
    ),
    TestCase(
        id="reason_004",
        category=TestCategory.REASONING,
        prompt="Tom is looking at Mary. Mary is looking at John. Tom is married. John is not married. Is a married person looking at an unmarried person? Answer 'yes', 'no', or 'cannot be determined'.",
        expected="yes",
        scoring_method=ScoringMethod.RUBRIC,
        description="Mary's marital status puzzle",
        difficulty=4,
        tags=["logic puzzle"],
    ),
    TestCase(
        id="reason_005",
        category=TestCategory.REASONING,
        prompt="A bat and ball cost $1.10 together. The bat costs $1 more than the ball. How much does the ball cost in cents? Reply with just the number.",
        expected="5",
        scoring_method=ScoringMethod.NUMERIC,
        description="Classic bat-and-ball problem",
        difficulty=3,
        tags=["trick question", "famous"],
    ),
    TestCase(
        id="reason_006",
        category=TestCategory.REASONING,
        prompt="If you overtake the person in 2nd place in a race, what place are you in? Reply with just the number.",
        expected="2",
        scoring_method=ScoringMethod.EXACT,
        description="Race position puzzle",
        difficulty=2,
        tags=["trick question"],
    ),
    TestCase(
        id="reason_007",
        category=TestCategory.REASONING,
        prompt="A farmer has 17 sheep. All but 9 die. How many are left? Reply with just the number.",
        expected="9",
        scoring_method=ScoringMethod.EXACT,
        description="'All but' phrasing puzzle",
        difficulty=2,
        tags=["trick question"],
    ),
    TestCase(
        id="reason_008",
        category=TestCategory.REASONING,
        prompt="Which is heavier: a pound of feathers or a pound of steel? Answer 'feathers', 'steel', or 'same'.",
        expected="same",
        scoring_method=ScoringMethod.RUBRIC,
        description="Pound comparison trick",
        difficulty=1,
        tags=["trick question", "famous"],
    ),
    TestCase(
        id="reason_009",
        category=TestCategory.REASONING,
        prompt="If there are 3 apples and you take away 2, how many apples do you have? Reply with just the number.",
        expected="2",
        scoring_method=ScoringMethod.EXACT,
        description="'You have' vs 'remaining' puzzle",
        difficulty=2,
        tags=["trick question"],
    ),
    TestCase(
        id="reason_010",
        category=TestCategory.REASONING,
        prompt="A doctor gives you 3 pills and tells you to take one every half hour. How many minutes will the pills last? Reply with just the number.",
        expected="60",
        scoring_method=ScoringMethod.NUMERIC,
        description="Time interval puzzle",
        difficulty=2,
        tags=["trick question"],
    ),
]


# =============================================================================
# FACTUAL TESTS
# =============================================================================
# These test knowledge recall. The answers should never change.
# Good for detecting if a model's knowledge base has been altered.

FACTUAL_TESTS = [
    TestCase(
        id="fact_001",
        category=TestCategory.FACTUAL,
        prompt="What is the capital of France? Reply with just the city name.",
        expected="Paris",
        scoring_method=ScoringMethod.EXACT,
        description="Geography - France capital",
        difficulty=1,
    ),
    TestCase(
        id="fact_002",
        category=TestCategory.FACTUAL,
        prompt="Who wrote Romeo and Juliet? Reply with just the author's last name.",
        expected="Shakespeare",
        scoring_method=ScoringMethod.EXACT,
        description="Literature - Shakespeare",
        difficulty=1,
    ),
    TestCase(
        id="fact_003",
        category=TestCategory.FACTUAL,
        prompt="What planet is known as the Red Planet? Reply with just the planet name.",
        expected="Mars",
        scoring_method=ScoringMethod.EXACT,
        description="Astronomy - Mars",
        difficulty=1,
    ),
    TestCase(
        id="fact_004",
        category=TestCategory.FACTUAL,
        prompt="In what year did World War II end? Reply with just the year.",
        expected="1945",
        scoring_method=ScoringMethod.EXACT,
        description="History - WWII end",
        difficulty=1,
    ),
    TestCase(
        id="fact_005",
        category=TestCategory.FACTUAL,
        prompt="What is the chemical symbol for gold? Reply with just the symbol.",
        expected="Au",
        scoring_method=ScoringMethod.EXACT,
        description="Chemistry - Gold symbol",
        difficulty=1,
    ),
    TestCase(
        id="fact_006",
        category=TestCategory.FACTUAL,
        prompt="Who painted the Mona Lisa? Reply with the artist's full name.",
        expected="Leonardo da Vinci",
        scoring_method=ScoringMethod.EXACT,
        description="Art - Mona Lisa artist",
        difficulty=1,
    ),
    TestCase(
        id="fact_007",
        category=TestCategory.FACTUAL,
        prompt="What is the largest ocean on Earth? Reply with just the ocean name.",
        expected="Pacific",
        scoring_method=ScoringMethod.EXACT,
        description="Geography - Largest ocean",
        difficulty=1,
    ),
    TestCase(
        id="fact_008",
        category=TestCategory.FACTUAL,
        prompt="How many continents are there on Earth? Reply with just the number.",
        expected="7",
        scoring_method=ScoringMethod.EXACT,
        description="Geography - Number of continents",
        difficulty=1,
    ),
    TestCase(
        id="fact_009",
        category=TestCategory.FACTUAL,
        prompt="What is the freezing point of water in Celsius? Reply with just the number.",
        expected="0",
        scoring_method=ScoringMethod.EXACT,
        description="Science - Water freezing point",
        difficulty=1,
    ),
    TestCase(
        id="fact_010",
        category=TestCategory.FACTUAL,
        prompt="Who was the first person to walk on the moon? Reply with just their full name.",
        expected="Neil Armstrong",
        scoring_method=ScoringMethod.EXACT,
        description="History - Moon landing",
        difficulty=1,
    ),
]


# =============================================================================
# CONSISTENCY TESTS
# =============================================================================
# Same question asked in different ways. Model should give the same answer.
# Groups of 3 questions each (a, b, c) that should have identical answers.

CONSISTENCY_TESTS = [
    # Group 1: Simple addition
    TestCase(
        id="consist_001a",
        category=TestCategory.CONSISTENCY,
        prompt="What is 25 + 37? Reply with just the number.",
        expected="62",
        scoring_method=ScoringMethod.EXACT,
        description="Addition - phrasing A",
        tags=["group_001"],
    ),
    TestCase(
        id="consist_001b",
        category=TestCategory.CONSISTENCY,
        prompt="Calculate: 25 plus 37. Reply with just the number.",
        expected="62",
        scoring_method=ScoringMethod.EXACT,
        description="Addition - phrasing B",
        tags=["group_001"],
    ),
    TestCase(
        id="consist_001c",
        category=TestCategory.CONSISTENCY,
        prompt="25 + 37 = ? Reply with just the number.",
        expected="62",
        scoring_method=ScoringMethod.EXACT,
        description="Addition - phrasing C",
        tags=["group_001"],
    ),
    
    # Group 2: Capital city
    TestCase(
        id="consist_002a",
        category=TestCategory.CONSISTENCY,
        prompt="What is the capital of Japan? Reply with just the city name.",
        expected="Tokyo",
        scoring_method=ScoringMethod.EXACT,
        description="Japan capital - phrasing A",
        tags=["group_002"],
    ),
    TestCase(
        id="consist_002b",
        category=TestCategory.CONSISTENCY,
        prompt="Name the capital city of Japan. Reply with just the city name.",
        expected="Tokyo",
        scoring_method=ScoringMethod.EXACT,
        description="Japan capital - phrasing B",
        tags=["group_002"],
    ),
    TestCase(
        id="consist_002c",
        category=TestCategory.CONSISTENCY,
        prompt="Japan's capital is which city? Reply with just the city name.",
        expected="Tokyo",
        scoring_method=ScoringMethod.EXACT,
        description="Japan capital - phrasing C",
        tags=["group_002"],
    ),
    
    # Group 3: Prime number
    TestCase(
        id="consist_003a",
        category=TestCategory.CONSISTENCY,
        prompt="Is 17 a prime number? Answer only 'yes' or 'no'.",
        expected="yes",
        scoring_method=ScoringMethod.EXACT,
        description="Prime check - phrasing A",
        tags=["group_003"],
    ),
    TestCase(
        id="consist_003b",
        category=TestCategory.CONSISTENCY,
        prompt="Tell me if 17 is prime. Answer only 'yes' or 'no'.",
        expected="yes",
        scoring_method=ScoringMethod.EXACT,
        description="Prime check - phrasing B",
        tags=["group_003"],
    ),
    TestCase(
        id="consist_003c",
        category=TestCategory.CONSISTENCY,
        prompt="Prime number check: 17. Answer only 'yes' or 'no'.",
        expected="yes",
        scoring_method=ScoringMethod.EXACT,
        description="Prime check - phrasing C",
        tags=["group_003"],
    ),
]


# =============================================================================
# INSTRUCTION-FOLLOWING TESTS
# =============================================================================
# Tests whether the model follows format constraints.
# Important for applications where output format matters.

INSTRUCTION_TESTS = [
    TestCase(
        id="instr_001",
        category=TestCategory.INSTRUCTION,
        prompt="List exactly 3 colors. No more, no less. One per line.",
        expected=None,  # Checked by format scoring
        scoring_method=ScoringMethod.FORMAT,
        description="List exactly 3 items",
        difficulty=1,
    ),
    TestCase(
        id="instr_002",
        category=TestCategory.INSTRUCTION,
        prompt="Answer with only a single word: What color is the sky on a clear day?",
        expected=None,
        scoring_method=ScoringMethod.FORMAT,
        description="Single word response",
        difficulty=1,
    ),
    TestCase(
        id="instr_003",
        category=TestCategory.INSTRUCTION,
        prompt="Respond with exactly 5 words. No more, no less.",
        expected=None,
        scoring_method=ScoringMethod.FORMAT,
        description="Exactly 5 words",
        difficulty=2,
    ),
    TestCase(
        id="instr_004",
        category=TestCategory.INSTRUCTION,
        prompt="Give me a number between 1 and 10. Just the digit, nothing else.",
        expected=None,
        scoring_method=ScoringMethod.FORMAT,
        description="Just a digit 1-10",
        difficulty=1,
    ),
    TestCase(
        id="instr_005",
        category=TestCategory.INSTRUCTION,
        prompt="Write a sentence with exactly 4 words.",
        expected=None,
        scoring_method=ScoringMethod.FORMAT,
        description="Exactly 4 word sentence",
        difficulty=2,
    ),
    TestCase(
        id="instr_006",
        category=TestCategory.INSTRUCTION,
        prompt="Answer in ALL CAPS: What is 2+2?",
        expected=None,
        scoring_method=ScoringMethod.FORMAT,
        description="All caps response",
        difficulty=1,
    ),
    TestCase(
        id="instr_007",
        category=TestCategory.INSTRUCTION,
        prompt="Reply with only 'yes' or 'no': Is water wet?",
        expected=None,
        scoring_method=ScoringMethod.FORMAT,
        description="Binary yes/no response",
        difficulty=1,
    ),
    TestCase(
        id="instr_008",
        category=TestCategory.INSTRUCTION,
        prompt="List 5 numbers in descending order from 10 to 6, separated by commas.",
        expected=None,
        scoring_method=ScoringMethod.FORMAT,
        description="Formatted number list",
        difficulty=2,
    ),
]


# =============================================================================
# FULL TEST SUITE
# =============================================================================

def get_all_tests() -> List[TestCase]:
    """
    Return all test cases.
    
    Returns:
        List of all TestCase objects
    """
    return (
        MATH_TESTS + 
        REASONING_TESTS + 
        FACTUAL_TESTS + 
        CONSISTENCY_TESTS + 
        INSTRUCTION_TESTS +
        get_all_new_tests()  # NEW
    )


def get_tests_by_category(category: TestCategory) -> List[TestCase]:
    """
    Get tests filtered by category.
    
    Args:
        category: The TestCategory to filter by
    
    Returns:
        List of TestCase objects in that category
    """
    return [t for t in get_all_tests() if t.category == category]


def get_test_by_id(test_id: str) -> Optional[TestCase]:
    """
    Get a specific test by its ID.
    
    Args:
        test_id: The test ID (e.g., "math_001")
    
    Returns:
        The TestCase if found, None otherwise
    """
    for test in get_all_tests():
        if test.id == test_id:
            return test
    return None


# Quick summary when this file is run directly
if __name__ == "__main__":
    all_tests = get_all_tests()
    print(f"Total tests: {len(all_tests)}")
    print("\nBy category:")
    for cat in TestCategory:
        count = len(get_tests_by_category(cat))
        print(f"  {cat.value}: {count}")