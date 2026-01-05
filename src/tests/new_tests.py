"""Creative writing, coding, and advanced reasoning test cases."""

from typing import List
from dataclasses import dataclass
from typing import Optional


# Define TestCase here to avoid circular import
@dataclass
class TestCase:
    id: str
    category: str
    prompt: str
    expected: Optional[str]
    scoring_method: str  # CHANGED from scoring_type
    metadata: dict = None


# Creative Writing Tests
CREATIVE_TESTS = [
    TestCase(
        id="creative_001",
        category="creative",
        prompt="Write exactly 3 sentences telling a scary story. Keep it under 50 words total.",
        expected=None,
        scoring_method="creative_structure",
        metadata={"expected_sentences": 3, "max_words": 50}
    ),
    TestCase(
        id="creative_002",
        category="creative",
        prompt="Write a haiku about programming. Format: line1 / line2 / line3",
        expected=None,
        scoring_method="creative_structure",
        metadata={"expected_lines": 3}
    ),
    TestCase(
        id="creative_003",
        category="creative",
        prompt="Create a product slogan for AI-powered sneakers. Use exactly 5 words.",
        expected=None,
        scoring_method="creative_structure",
        metadata={"expected_words": 5}
    ),
    TestCase(
        id="creative_004",
        category="creative",
        prompt="Tell a joke in exactly 2 sentences.",
        expected=None,
        scoring_method="creative_structure",
        metadata={"expected_sentences": 2}
    ),
    TestCase(
        id="creative_005",
        category="creative",
        prompt="Describe a sunset in exactly 20 words.",
        expected=None,
        scoring_method="creative_structure",
        metadata={"expected_words": 20}
    ),
]

# Coding Tests
CODING_TESTS = [
    TestCase(
        id="code_001",
        category="code",
        prompt="Write a Python function called is_palindrome that checks if a string is a palindrome. Just the function, no explanation.",
        expected=None,
        scoring_method="code_check",
        metadata={"keywords": ["def", "is_palindrome", "[::-1]", "reverse"]}
    ),
    TestCase(
        id="code_002",
        category="code",
        prompt="What's wrong with this code?\nmy_list = [1, 2, 3]\nfor item in my_list:\n    my_list.remove(item)",
        expected=None,
        scoring_method="code_check",
        metadata={"keywords": ["modif", "iteration", "loop"]}
    ),
    TestCase(
        id="code_003",
        category="code",
        prompt="Write a Python list comprehension that gets all even numbers from 1 to 10.",
        expected=None,
        scoring_method="code_check",
        metadata={"keywords": ["[", "for", "if", "% 2"]}
    ),
    TestCase(
        id="code_004",
        category="code",
        prompt="What does this lambda do? lambda s: s[::-1]",
        expected=None,
        scoring_method="code_check",
        metadata={"keywords": ["reverse", "backward"]}
    ),
    TestCase(
        id="code_005",
        category="code",
        prompt="Write a SQL query to get the top 3 highest salaries from an employees table.",
        expected=None,
        scoring_method="code_check",
        metadata={"keywords": ["SELECT", "ORDER BY", "LIMIT 3", "TOP 3"]}
    ),
]

# Advanced Reasoning Tests
ADVANCED_REASONING_TESTS = [
    TestCase(
        id="reason_adv_001",
        category="reasoning",
        prompt="If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly?",
        expected="no",
        scoring_method="exact",
        metadata={"explanation": "This is invalid logic - we can't conclude anything about roses specifically"}
    ),
    TestCase(
        id="reason_adv_002",
        category="reasoning",
        prompt="You have a 3-gallon jug and a 5-gallon jug. How do you measure exactly 4 gallons?",
        expected=None,
        scoring_method="rubric",
        metadata={"concept": "Fill 5, pour into 3 (leaving 2), empty 3, pour 2 into 3, fill 5, pour into 3 = 4"}
    ),
    TestCase(
        id="reason_adv_003",
        category="reasoning",
        prompt="A farmer has 17 sheep. All but 9 die. How many are left?",
        expected="9",
        scoring_method="numeric",
        metadata={"trick": "all but 9 means 9 survived"}
    ),
    TestCase(
        id="reason_adv_004",
        category="reasoning",
        prompt="If you're running a race and you pass the person in 2nd place, what place are you in?",
        expected="2",
        scoring_method="exact",
        metadata={"explanation": "You're now in 2nd, not 1st"}
    ),
    TestCase(
        id="reason_adv_005",
        category="reasoning",
        prompt="A doctor gives you 3 pills and says to take one every 30 minutes. How long do the pills last?",
        expected="60",
        scoring_method="numeric",
        metadata={"explanation": "60 minutes total (0 min, 30 min, 60 min)"}
    ),
]

# Common Sense Tests
COMMON_SENSE_TESTS = [
    TestCase(
        id="common_001",
        category="reasoning",
        prompt="You're in a room with 2 doors. One leads to certain death, one to freedom. There are 2 guards - one always lies, one always tells truth. You can ask ONE question to ONE guard. What do you ask?",
        expected=None,
        scoring_method="rubric",
        metadata={"concept": "Ask either guard what the other would say, then choose opposite"}
    ),
    TestCase(
        id="common_002",
        category="reasoning",
        prompt="If it takes 5 machines 5 minutes to make 5 widgets, how long does it take 100 machines to make 100 widgets?",
        expected="5",
        scoring_method="numeric",
        metadata={"explanation": "Each machine makes 1 widget in 5 minutes"}
    ),
    TestCase(
        id="common_003",
        category="reasoning",
        prompt="A man pushes his car to a hotel and immediately knows he's bankrupt. Why?",
        expected=None,
        scoring_method="rubric",
        metadata={"answer": "He's playing Monopoly"}
    ),
]


def get_all_new_tests() -> List[TestCase]:
    """Return all new test cases."""
    return (
        CREATIVE_TESTS +
        CODING_TESTS +
        ADVANCED_REASONING_TESTS +
        COMMON_SENSE_TESTS
    )


if __name__ == "__main__":
    tests = get_all_new_tests()
    print(f"Total new tests: {len(tests)}")
    print(f"  Creative: {len(CREATIVE_TESTS)}")
    print(f"  Coding: {len(CODING_TESTS)}")
    print(f"  Advanced Reasoning: {len(ADVANCED_REASONING_TESTS)}")
    print(f"  Common Sense: {len(COMMON_SENSE_TESTS)}")