# tests/test_similarity_metrics.py

import sys
import os
import gensim
import numpy as np
import time

# Adjust the path to ensure the test script can access src modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
src_dir = os.path.join(parent_dir, 'src')
sys.path.append(src_dir)

# Import the similarity metrics functions and model loader
from similarity_metrics import (
    calculate_cosine_similarity,
    calculate_euclidean_similarity,
    calculate_neighbor_overlap,
    calculate_semantic_similarity,
    calculate_jaccard_similarity,
    ngram_jaccard_similarity,
    calculate_levenshtein_distance
)
from model_loader import load_model

def print_separator():
    print("-" * 80)

def run_test(test_name, func, args, expected, tolerance=0.05):
    """
    Runs a single test case.

    Parameters:
    - test_name (str): The name of the test.
    - func (callable): The similarity function to test.
    - args (tuple): Arguments to pass to the function.
    - expected (float or None or int): The expected result.
    - tolerance (float): Acceptable deviation for float comparisons.

    Returns:
    - bool: True if the test passes, False otherwise.
    """
    print(f"Running Test: {test_name}")
    try:
        result = func(*args)
    except Exception as e:
        print(f"❌ Failed: Exception occurred - {e}")
        return False

    if expected is None:
        if result is None:
            print("✅ Passed: Correctly handled out-of-vocabulary words.")
            return True
        else:
            print("❌ Failed: Did not handle out-of-vocabulary words as expected.")
            print(f"   Expected: {expected}, Got: {result}")
            return False
    else:
        if result is None:
            print("❌ Failed: Function returned None unexpectedly.")
            print(f"   Expected: {expected}, Got: {result}")
            return False
        if isinstance(expected, float):
            if abs(result - expected) <= tolerance:
                print(f"✅ Passed: Expected ≈{expected}, Got: {result:.4f}")
                return True
            else:
                print(f"❌ Failed: Expected ≈{expected}, Got: {result:.4f}")
                return False
        elif isinstance(expected, int):
            if result == expected:
                print(f"✅ Passed: Expected {expected}, Got: {result}")
                return True
            else:
                print(f"❌ Failed: Expected {expected}, Got: {result}")
                return False
        else:
            # For non-numeric expected results
            if result == expected:
                print(f"✅ Passed: Expected {expected}, Got: {result}")
                return True
            else:
                print(f"❌ Failed: Expected {expected}, Got: {result}")
                return False

def main():
    """
    Main function to run all test cases.
    """
    # Load the model
    print("Loading the FastText model...")
    model = load_model()
    print("Model loaded successfully.\n")

    # Define test cases
    test_cases = [
        # Cosine Similarity Tests
        {
            'test_name': 'Cosine Similarity - Similar Words',
            'func': calculate_cosine_similarity,
            'args': ('king', 'queen', model),
            'expected': 0.7,  # Expected similarity > 0.5
            'tolerance': 0.2
        },
        {
            'test_name': 'Cosine Similarity - Dissimilar Words',
            'func': calculate_cosine_similarity,
            'args': ('apple', 'car', model),
            'expected': 0.2,  # Expected similarity < 0.3
            'tolerance': 0.2
        },
        {
            'test_name': 'Cosine Similarity - Out-of-Vocabulary Word',
            'func': calculate_cosine_similarity,
            'args': ('apple', 'asdfghjkl', model),
            'expected': None
        },
        {
            'test_name': 'Cosine Similarity - Self Similarity',
            'func': calculate_cosine_similarity,
            'args': ('apple', 'apple', model),
            'expected': 1.0,
            'tolerance': 0.0
        },
        # Euclidean Similarity Tests
        {
            'test_name': 'Euclidean Similarity - Similar Words',
            'func': calculate_euclidean_similarity,
            'args': ('king', 'queen', model),
            'expected': 0.3,  # Expected similarity > 0.2
            'tolerance': 0.2
        },
        {
            'test_name': 'Euclidean Similarity - Dissimilar Words',
            'func': calculate_euclidean_similarity,
            'args': ('apple', 'car', model),
            'expected': 0.1,  # Expected similarity < 0.2
            'tolerance': 0.1
        },
        {
            'test_name': 'Euclidean Similarity - Out-of-Vocabulary Word',
            'func': calculate_euclidean_similarity,
            'args': ('apple', 'asdfghjkl', model),
            'expected': None
        },
        {
            'test_name': 'Euclidean Similarity - Self Similarity',
            'func': calculate_euclidean_similarity,
            'args': ('apple', 'apple', model),
            'expected': 1.0,
            'tolerance': 0.0
        },
        # Neighbor Overlap Similarity Tests
        {
            'test_name': 'Neighbor Overlap - Similar Words',
            'func': calculate_neighbor_overlap,
            'args': ('apple', 'banana', model, 50),
            'expected': 0.3,  # Expected overlap > 0.2
            'tolerance': 0.2
        },
        {
            'test_name': 'Neighbor Overlap - Dissimilar Words',
            'func': calculate_neighbor_overlap,
            'args': ('apple', 'car', model, 50),
            'expected': 0.05,  # Expected overlap < 0.1
            'tolerance': 0.05
        },
        {
            'test_name': 'Neighbor Overlap - Out-of-Vocabulary Word',
            'func': calculate_neighbor_overlap,
            'args': ('apple', 'asdfghjkl', model, 50),
            'expected': None
        },
        {
            'test_name': 'Neighbor Overlap - Self Similarity',
            'func': calculate_neighbor_overlap,
            'args': ('apple', 'apple', model, 50),
            'expected': 1.0,
            'tolerance': 0.0
        },
        # Combined Semantic Similarity Tests
        {
            'test_name': 'Semantic Similarity - Similar Words',
            'func': calculate_semantic_similarity,
            'args': ('king', 'queen', model, 50, {'cosine':0.4, 'euclidean':0.3, 'neighbor':0.3}),
            'expected': 0.7,  # Expected similarity > 0.5
            'tolerance': 0.3
        },
        {
            'test_name': 'Semantic Similarity - Dissimilar Words',
            'func': calculate_semantic_similarity,
            'args': ('apple', 'car', model, 50, {'cosine':0.4, 'euclidean':0.3, 'neighbor':0.3}),
            'expected': 0.2,  # Expected similarity < 0.3
            'tolerance': 0.2
        },
        {
            'test_name': 'Semantic Similarity - Out-of-Vocabulary Word',
            'func': calculate_semantic_similarity,
            'args': ('apple', 'asdfghjkl', model, 50, {'cosine':0.4, 'euclidean':0.3, 'neighbor':0.3}),
            'expected': 0.0
        },
        {
            'test_name': 'Semantic Similarity - Self Similarity',
            'func': calculate_semantic_similarity,
            'args': ('apple', 'apple', model, 50, {'cosine':0.4, 'euclidean':0.3, 'neighbor':0.3}),
            'expected': 1.0,
            'tolerance': 0.0
        },
        # Jaccard Similarity Tests
        {
            'test_name': 'Jaccard Similarity - Similar Words',
            'func': calculate_jaccard_similarity,
            'args': ('apple', 'apples'),
            'expected': 0.8,  # Expected high similarity
            'tolerance': 0.2
        },
        {
            'test_name': 'Jaccard Similarity - Dissimilar Words',
            'func': calculate_jaccard_similarity,
            'args': ('apple', 'car'),
            'expected': 0.0,  # Expected low similarity
            'tolerance': 0.1
        },
        {
            'test_name': 'Jaccard Similarity - Identical Words',
            'func': calculate_jaccard_similarity,
            'args': ('apple', 'apple'),
            'expected': 1.0,
            'tolerance': 0.0
        },
        {
            'test_name': 'Jaccard Similarity - Completely Different Words',
            'func': calculate_jaccard_similarity,
            'args': ('abc', 'xyz'),
            'expected': 0.0,
            'tolerance': 0.0
        },
        {
            'test_name': 'Jaccard Similarity - Empty Strings',
            'func': calculate_jaccard_similarity,
            'args': ('', ''),
            'expected': 0.0
        },
        # N-gram Jaccard Similarity Tests
        {
            'test_name': 'N-gram Jaccard Similarity - Similar Words',
            'func': ngram_jaccard_similarity,
            'args': ('apple', 'apples'),
            'expected': 0.8,  # Expected high similarity
            'tolerance': 0.2
        },
        {
            'test_name': 'N-gram Jaccard Similarity - Dissimilar Words',
            'func': ngram_jaccard_similarity,
            'args': ('apple', 'car'),
            'expected': 0.0,  # Expected low similarity
            'tolerance': 0.1
        },
        {
            'test_name': 'N-gram Jaccard Similarity - Identical Words',
            'func': ngram_jaccard_similarity,
            'args': ('apple', 'apple'),
            'expected': 1.0,
            'tolerance': 0.0
        },
        {
            'test_name': 'N-gram Jaccard Similarity - Completely Different Words',
            'func': ngram_jaccard_similarity,
            'args': ('abc', 'xyz'),
            'expected': 0.0,
            'tolerance': 0.0
        },
        {
            'test_name': 'N-gram Jaccard Similarity - Empty Strings',
            'func': ngram_jaccard_similarity,
            'args': ('', ''),
            'expected': 0.0
        },
        # Levenshtein Distance Tests
        {
            'test_name': 'Levenshtein Distance - Similar Words',
            'func': calculate_levenshtein_distance,
            'args': ('kitten', 'sitting'),
            'expected': 3  # Expected distance
        },
        {
            'test_name': 'Levenshtein Distance - Dissimilar Words',
            'func': calculate_levenshtein_distance,
            'args': ('apple', 'car'),
            'expected': 3  # Expected distance
        },
        {
            'test_name': 'Levenshtein Distance - Identical Words',
            'func': calculate_levenshtein_distance,
            'args': ('apple', 'apple'),
            'expected': 0
        },
        {
            'test_name': 'Levenshtein Distance - Completely Different Words',
            'func': calculate_levenshtein_distance,
            'args': ('abc', 'xyz'),
            'expected': 3
        },
        {
            'test_name': 'Levenshtein Distance - One Empty String',
            'func': calculate_levenshtein_distance,
            'args': ('apple', ''),
            'expected': 5
        },
        {
            'test_name': 'Levenshtein Distance - Both Empty Strings',
            'func': calculate_levenshtein_distance,
            'args': ('', ''),
            'expected': 0
        },
        {
            'test_name': 'Levenshtein Distance - Single Character Difference',
            'func': calculate_levenshtein_distance,
            'args': ('a', 'b'),
            'expected': 1
        },
    ]

    # Initialize counters
    total_tests = len(test_cases)
    passed_tests = 0

    print_separator()
    print("Starting Similarity Metrics Tests")
    print_separator()

    for test in test_cases:
        result = run_test(
            test_name=test['test_name'],
            func=test['func'],
            args=test['args'],
            expected=test['expected'],
            tolerance=test.get('tolerance', 0.05)  # Default tolerance if not specified
        )
        if result:
            passed_tests += 1
        print_separator()

    # Summary of Test Results
    print("\nTest Summary")
    print_separator()
    print(f"Total Tests Run: {total_tests}")
    print(f"Tests Passed: {passed_tests}")
    print(f"Tests Failed: {total_tests - passed_tests}")

    if passed_tests == total_tests:
        print("🎉 All tests passed successfully!")
    else:
        print("⚠️ Some tests failed. Please review the failed test cases.")

def run_test(test_name, func, args, expected, tolerance=0.05):
    """
    Runs a single test case.

    Parameters:
    - test_name (str): The name of the test.
    - func (callable): The similarity function to test.
    - args (tuple): Arguments to pass to the function.
    - expected (float or None or int): The expected result.
    - tolerance (float): Acceptable deviation for float comparisons.

    Returns:
    - bool: True if the test passes, False otherwise.
    """
    print(f"Running Test: {test_name}")
    try:
        result = func(*args)
    except Exception as e:
        print(f"❌ Failed: Exception occurred - {e}")
        return False

    if expected is None:
        if result is None:
            print("✅ Passed: Correctly handled out-of-vocabulary words.")
            return True
        else:
            print("❌ Failed: Did not handle out-of-vocabulary words as expected.")
            print(f"   Expected: {expected}, Got: {result}")
            return False
    else:
        if result is None:
            print("❌ Failed: Function returned None unexpectedly.")
            print(f"   Expected: {expected}, Got: {result}")
            return False
        if isinstance(expected, float):
            if abs(result - expected) <= tolerance:
                print(f"✅ Passed: Expected ≈{expected}, Got: {result:.4f}")
                return True
            else:
                print(f"❌ Failed: Expected ≈{expected}, Got: {result:.4f}")
                return False
        elif isinstance(expected, int):
            if result == expected:
                print(f"✅ Passed: Expected {expected}, Got: {result}")
                return True
            else:
                print(f"❌ Failed: Expected {expected}, Got: {result}")
                return False
        else:
            # For non-numeric expected results
            if result == expected:
                print(f"✅ Passed: Expected {expected}, Got: {result}")
                return True
            else:
                print(f"❌ Failed: Expected {expected}, Got: {result}")
                return False

if __name__ == "__main__":
    main()
