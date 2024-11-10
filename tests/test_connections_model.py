# test_connections_model.py

import sys
import os

# Adjust the path to ensure the test script can access src modules
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, os.pardir, 'src'))
sys.path.append(src_dir)

from model_loader import ModelLoader
from connections_model import connections_model
from config import DATA_PATH

def test_connections_model():
    model_instance = ModelLoader.load_vectors()  # Defaults to 'embeddings/fasttext_vectors.kv'

    # Sample words to group (single words only)
    words = [
        'apple', 'banana', 'cherry', 'date',       # Expected Group 1: Fruits
        'dog', 'cat', 'mouse', 'rabbit',           # Expected Group 2: Animals
        'red', 'blue', 'green', 'yellow',          # Expected Group 3: Colors
        'car', 'bus', 'train', 'plane'             # Expected Group 4: Transportation
    ]

    # Expected groups for comparison
    expected_groups = {
        'Group1': ['apple', 'banana', 'cherry', 'date'],
        'Group2': ['dog', 'cat', 'mouse', 'rabbit'],
        'Group3': ['red', 'blue', 'green', 'yellow'],
        'Group4': ['car', 'bus', 'train', 'plane']
    }

    # Call the connections_model function
    groups = connections_model(words, model_instance)

    # Print the expected groups
    print("\nExpected Groups:")
    for group_name, group_words in expected_groups.items():
        print(f"{group_name}: {group_words}")

    # Print the groups generated by connections_model
    print("\nGroups generated by connections_model:")
    for group_name, group_words in groups.items():
        print(f"{group_name}: {group_words}")

    # Compare generated groups with expected groups
    correct_groups = 0
    for expected_group in expected_groups.values():
        if any(set(expected_group) == set(generated_group) for generated_group in groups.values()):
            correct_groups += 1

    print(f"\nNumber of correctly formed groups: {correct_groups} out of {len(expected_groups)}")

if __name__ == "__main__":
    test_connections_model()
