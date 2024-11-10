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
    calculate_neighbor_overlap,
    calculate_euclidean_similarity,
    ngram_jaccard_similarity,
    calculate_levenshtein_distance,
    calculate_semantic_similarity
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
    # Semantic
    EUCLEDIAN_THRESHOLD = 0.1 
    COSINE_THRESHOLD = 0.1
    SEMANTIC_SIMILARITY_THRESHOLD = 0.5 # Needs to be fine-tuned.  

    # Lexical
    JACCARD_THRESHOLD = 0.1  # Lowered threshold for lexical similarity

    # Spelling
    LEVENSHTEIN_THRESHOLD = 10  # Increased threshold for spelling similarity

    # Weights for the similarity components
    weights = {'cosine': 0.4, 'euclidean': 0.3, 'neighbor': 0.3}


    # Initialize data structures
    groups = defaultdict(list)  # Dictionary to hold groups of words
    used_words = set()  # Set to keep track of words that have already been grouped

    # Step 1: Group words based on semantic similarity (cosine similarity)
    print("Step 1: Semantic Similarity Grouping")
    for word1 in words:
        if word1 in used_words:
            continue  # Skip words that have already been grouped
        group = [word1]  # Start a new group with the current word
        used_words.add(word1)
        print(f"Creating new group with seed word: {word1}")
        for word2 in words:
            if word2 not in used_words and word2 != word1:
                # Check similarity with any member of the group
                semantic_similarities = [calculate_semantic_similarity(w, word2, model,top_n=50,weights=weights) for w in group]
                max_similarity = max(semantic_similarities)
                print(f"Checking word: {word2}")
                print(f"  Semantic similarities with group members: {list(zip(group, semantic_similarities))}")
                if max_similarity >= SEMANTIC_SIMILARITY_THRESHOLD:
                    group.append(word2)
                    used_words.add(word2)
                    print(f"  Added {word2} to group (max similarity: {max_similarity})")
                else:
                    print(f"  Did not add {word2} (max similarity: {max_similarity})")
                if len(group) == 4:
                    break  # Stop adding words if the group reaches four words
        if len(group) == 4:
            # Add the group to the groups dictionary
            group_name = f"Group{len(groups) + 1}"
            groups[group_name] = group
            print(f"Formed group {group_name}: {group}")
        else:
            print(f"Could not form a full group with seed word: {word1}")
            used_words.difference_update(group)  # Remove words if group is incomplete

    # Step 2: Group remaining words based on lexical similarity (Jaccard similarity)
    print("\nStep 2: Lexical Similarity Grouping")
    remaining_words = [word for word in words if word not in used_words]
    for word1 in remaining_words:
        if word1 in used_words:
            continue
        group = [word1]
        used_words.add(word1)
        print(f"Creating new group with seed word: {word1}")
        for word2 in remaining_words:
            if word2 not in used_words and word2 != word1:
                # Check similarity with any member of the group
                similarities = [ngram_jaccard_similarity(w, word2,n=2) for w in group]
                max_similarity = max(similarities)
                print(f"Checking word: {word2}")
                print(f"  Similarities with group members: {list(zip(group, similarities))}")
                if max_similarity >= JACCARD_THRESHOLD:
                    group.append(word2)
                    used_words.add(word2)
                    print(f"  Added {word2} to group (max similarity: {max_similarity})")
                else:
                    print(f"  Did not add {word2} (max similarity: {max_similarity})")
                if len(group) == 4:
                    break
        if len(group) == 4:
            group_name = f"Group{len(groups) + 1}"
            groups[group_name] = group
            print(f"Formed group {group_name}: {group}")
        else:
            print(f"Could not form a full group with seed word: {word1}")
            used_words.difference_update(group)

    # Step 3: Group remaining words based on spelling similarity (Levenshtein distance)
    print("\nStep 3: Spelling Similarity Grouping")
    remaining_words = [word for word in words if word not in used_words]
    for word1 in remaining_words:
        if word1 in used_words:
            continue
        group = [word1]
        used_words.add(word1)
        print(f"Creating new group with seed word: {word1}")
        for word2 in remaining_words:
            if word2 not in used_words and word2 != word1:
                # Check similarity with any member of the group
                distances = [calculate_levenshtein_distance(w, word2) for w in group]
                min_distance = min(distances)
                print(f"Checking word: {word2}")
                print(f"  Distances with group members: {list(zip(group, distances))}")
                if min_distance <= LEVENSHTEIN_THRESHOLD:
                    group.append(word2)
                    used_words.add(word2)
                    print(f"  Added {word2} to group (min distance: {min_distance})")
                else:
                    print(f"  Did not add {word2} (min distance: {min_distance})")
                if len(group) == 4:
                    break
        if len(group) == 4:
            group_name = f"Group{len(groups) + 1}"
            groups[group_name] = group
            print(f"Formed group {group_name}: {group}")
        else:
            print(f"Could not form a full group with seed word: {word1}")
            used_words.difference_update(group)

    # Final grouping of remaining words to ensure all words are grouped
    remaining_words = [word for word in words if word not in used_words]
    if remaining_words:
        print("\nFinal grouping of remaining words")
        while remaining_words:
            group = remaining_words[:4]
            group_name = f"Group{len(groups) + 1}"
            groups[group_name] = group
            print(f"Formed group {group_name}: {group}")
            used_words.update(group)
            remaining_words = remaining_words[4:]

    return groups
