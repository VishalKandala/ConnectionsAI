# src/connections_model.py

###############################################################################
#                                                                             #
#                             Connections Model                               #
#                                                                             #
#      A module to group words into categories based on similarity metrics.   #
#                                                                             #
###############################################################################

# Import necessary libraries
from collections import defaultdict  # For grouping words

# Import similarity functions
from .similarity_metrics import (
    calculate_cosine_similarity,
    calculate_jaccard_similarity,
    calculate_levenshtein_distance
)

def connections_model(words, model):
    """
    Group words into categories based on semantic, lexical, and spelling similarities.

    Parameters:
    - words (list): A list of words to be grouped.
    - model: The pre-trained FastText model.

    Returns:
    - dict: A dictionary where each key is a group name and each value is a list of words in that group.
    """

    # Similarity thresholds for grouping
    COSINE_THRESHOLD = 0.7  # Threshold for semantic similarity (cosine similarity)
    JACCARD_THRESHOLD = 0.8  # Threshold for lexical similarity (Jaccard similarity)
    LEVENSHTEIN_THRESHOLD = 3  # Threshold for spelling similarity (Levenshtein distance)

    # Initialize data structures
    groups = defaultdict(list)  # Dictionary to hold groups of words
    used_words = set()  # Set to keep track of words that have already been grouped

    # Step 1: Group words based on high semantic similarity (cosine similarity)
    for word1 in words:
        if word1 in used_words:
            continue  # Skip words that have already been grouped
        group = [word1]  # Start a new group with the current word
        used_words.add(word1)  # Mark the word as used
        for word2 in words:
            if word2 not in used_words:
                cosine_similarity = calculate_cosine_similarity(word1, word2, model)
                if cosine_similarity >= COSINE_THRESHOLD:
                    # Add word to group if it has high cosine similarity with word1
                    group.append(word2)
                    used_words.add(word2)  # Mark the word as used
            if len(group) == 4:
                break  # Stop adding words if the group reaches four words
        if len(group) == 4:
            # Add the group to the groups dictionary
            group_name = f"Group{len(groups) + 1}"
            groups[group_name] = group
        else:
            # Remove words from used_words if the group is incomplete
            used_words.difference_update(group)
    
    # Step 2: Group remaining words based on high lexical similarity (Jaccard similarity)
    remaining_words = [word for word in words if word not in used_words]
    for word1 in remaining_words:
        if word1 in used_words:
            continue
        group = [word1]
        used_words.add(word1)
        for word2 in remaining_words:
            if word2 not in used_words:
                jaccard_similarity = calculate_jaccard_similarity(word1, word2)
                if jaccard_similarity >= JACCARD_THRESHOLD:
                    group.append(word2)
                    used_words.add(word2)
            if len(group) == 4:
                break
        if len(group) == 4:
            group_name = f"Group{len(groups) + 1}"
            groups[group_name] = group
        else:
            used_words.difference_update(group)

    # Step 3: Group remaining words based on low spelling difference (Levenshtein distance)
    remaining_words = [word for word in words if word not in used_words]
    for word1 in remaining_words:
        if word1 in used_words:
            continue
        group = [word1]
        used_words.add(word1)
        for word2 in remaining_words:
            if word2 not in used_words:
                levenshtein_distance = calculate_levenshtein_distance(word1, word2)
                if levenshtein_distance <= LEVENSHTEIN_THRESHOLD:
                    group.append(word2)
                    used_words.add(word2)
            if len(group) == 4:
                break
        if len(group) == 4:
            group_name = f"Group{len(groups) + 1}"
            groups[group_name] = group
        else:
            used_words.difference_update(group)

    # Return the final groups of words
    return groups
