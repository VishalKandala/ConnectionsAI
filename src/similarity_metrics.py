# src/similarity_metrics.py

###############################################################################
#                                                                             #
#                           Similarity Metrics                                #
#                                                                             #
#      A module containing functions to compute various similarity metrics.   #
#                                                                             #
###############################################################################

# Import necessary libraries
import numpy as np  # For numerical computations
import Levenshtein  # For computing Levenshtein distance

def calculate_cosine_similarity(word1, word2, model):
    """
    Calculate the cosine similarity between two words using gensim FastText embeddings.

    Parameters:
    - word1 (str): The first word.
    - word2 (str): The second word.
    - model: The pre-trained FastText model.

    Returns:
    - float: The cosine similarity score between the two word vectors.
    """
    # Check if words are in the model's vocabulary
    if word1 in model.key_to_index and word2 in model.key_to_index:
        # Retrieve word vectors from the gensim model
        vec1 = model[word1]
        vec2 = model[word2]
        # Compute cosine similarity
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return similarity
    else:
        # Handle out-of-vocabulary words
        return 0.0  # Assign minimal similarity

def calculate_jaccard_similarity(word1, word2):
    """
    Calculate the Jaccard similarity between the character sets of two words.

    Parameters:
    - word1 (str): The first word.
    - word2 (str): The second word.

    Returns:
    - float: The Jaccard similarity score between the character sets of the two words.
    """
    # Convert words to sets of characters
    set1 = set(word1)
    set2 = set(word2)
    # Calculate intersection and union of the character sets
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    # Compute Jaccard similarity
    return intersection / union if union != 0 else 0

def calculate_levenshtein_distance(word1, word2):
    """
    Calculate the Levenshtein distance between two words.

    Parameters:
    - word1 (str): The first word.
    - word2 (str): The second word.

    Returns:
    - int: The Levenshtein distance between the two words.
    """
    # Compute Levenshtein distance using the Levenshtein library
    return Levenshtein.distance(word1, word2)
