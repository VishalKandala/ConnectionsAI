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
import time  # For profiling
def calculate_semantic_similarity(word1, word2, model, top_n=50, weights=None):
    """
    Calculate the combined semantic similarity between two words.
    
    Parameters:
    - word1 (str): The first word.
    - word2 (str): The second word.
    - model: The pre-trained word embedding model.
    - top_n (int): The number of neighbors to consider for neighbor overlap.
    - weights (dict): The weights for each similarity component.
    
    Returns:
    - float: The combined semantic similarity score.
    """
    if weights is None:
        weights = {'cosine': 0.4, 'euclidean': 0.3, 'neighbor': 0.3}
    
    # Ensure the weights sum to 1
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}
    
    # Initialize similarities
    cosine_sim = 0.0
    euclidean_sim = 0.0
    neighbor_sim = 0.0
    
    # Compute Cosine Similarity
    cosine_sim = calculate_cosine_similarity(word1, word2, model)
    if cosine_sim is None:
        cosine_sim = 0.0
    else:
        cosine_sim = (cosine_sim + 1) / 2  # Normalize to [0,1]
    
    # Compute Euclidean Similarity
    euclidean_dist = calculate_euclidean_similarity(word1, word2, model)
    if euclidean_dist is None:
        euclidean_sim = 0.0
    else:
        euclidean_sim = 1 / (1 + euclidean_dist)  # Similarity in (0,1]
    
    # Compute Neighbor Overlap Similarity
    neighbor_sim = calculate_neighbor_overlap(word1, word2, model, top_n)
    if neighbor_sim is None:
        neighbor_sim = 0.0
    
    # Combine similarities with weights
    combined_similarity = (
        weights['cosine'] * cosine_sim +
        weights['euclidean'] * euclidean_sim +
        weights['neighbor'] * neighbor_sim
    )
    
    return combined_similarity
   
def calculate_neighbor_overlap(word1, word2, model, top_n=50):
    """
    Calculate the neighbor overlap similarity between two words.

    Parameters:
    - word1 (str): The first word.
    - word2 (str): The second word.
    - model: The pre-trained word embedding model (e.g., FastText, Word2Vec).
    - top_n (int): The number of nearest neighbors to consider for each word.

    Returns:
    - float: The neighbor overlap similarity score between 0 and 1.
             Returns None if either word is not in the model's vocabulary.
    """
    # Start timing the function execution
    start_time = time.time()

    try:
        # Retrieve the top_n most similar words (neighbors) for word1
        neighbors1 = set()
        for neighbor, _ in model.most_similar(word1, topn=top_n):
            neighbors1.add(neighbor)

        # Retrieve the top_n most similar words (neighbors) for word2
        neighbors2 = set()
        for neighbor, _ in model.most_similar(word2, topn=top_n):
            neighbors2.add(neighbor)

        # Calculate the intersection of the neighbor sets
        overlap = neighbors1 & neighbors2

        # Calculate the neighbor overlap similarity score
        # Normalize the overlap by dividing by top_n to get a value between 0 and 1
        overlap_score = len(overlap) / top_n

        # End timing the function execution
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Profiling output: Print the overlap score and time taken
        print(f"Neighbor overlap between '{word1}' and '{word2}': {overlap_score:.4f}")
        print(f"Time taken: {elapsed_time:.4f} seconds")

        # Optional detailed output: Print the overlapping neighbors
        # print(f"Overlapping neighbors ({len(overlap)}): {sorted(overlap)}")

        return overlap_score

    except KeyError as e:
        # Handle the case where a word is not in the model's vocabulary
        print(f"Error: Word not in vocabulary - {e}")
        return None


def calculate_euclidean_similarity(word1, word2, model):
    """
    Calculate the Euclidean similarity between two words using gensim FastText embeddings.
    
    Parameters:
    - word1 (str): The first word.
    - word2 (str): The second word.
    - model: The pre-trained FastText model.
    
    Returns:
    - float: The Euclidean similarity between the two word vectors, normalized to [0,1].
             Returns None if either word is not in the model's vocabulary.
    """
    # Start timing the function execution
    start_time = time.time()
    
    try:
        # Check if both words are in the model's vocabulary
        if word1 in model.key_to_index and word2 in model.key_to_index:
            # Retrieve word vectors from the gensim model
            vec1 = model[word1]
            vec2 = model[word2]
            
            # Compute Euclidean distance
            distance = np.linalg.norm(vec1 - vec2)
            
            # Convert Euclidean distance to similarity score
            similarity = 1 / (1 + distance)  # Ensures similarity is between 0 and 1
            
            # End timing the function execution
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Profiling output: Print the similarity score and time taken
            print(f"Euclidean similarity between '{word1}' and '{word2}': {similarity:.4f}")
            print(f"Time taken: {elapsed_time:.6f} seconds")
            
            return similarity
        
        else:
            # Handle out-of-vocabulary words
            missing_words = [word for word in [word1, word2] if word not in model.key_to_index]
            print(f"Error: Word(s) not in vocabulary - {', '.join(missing_words)}")
            return None  # Indicates that similarity could not be computed
    
    except Exception as e:
        # Handle unexpected exceptions
        print(f"An error occurred: {e}")
        return None
    
import numpy as np
import time  # For profiling

def calculate_cosine_similarity(word1, word2, model):
    """
    Calculate the cosine similarity between two words using gensim FastText embeddings.
    
    Parameters:
    - word1 (str): The first word.
    - word2 (str): The second word.
    - model: The pre-trained FastText model.
    
    Returns:
    - float: The cosine similarity score between the two word vectors, normalized to [0,1].
             Returns None if either word is not in the model's vocabulary.
    """
    # Start timing the function execution
    start_time = time.time()
    
    try:
        # Check if both words are in the model's vocabulary
        if word1 in model.key_to_index and word2 in model.key_to_index:
            # Retrieve word vectors from the gensim model
            vec1 = model[word1]
            vec2 = model[word2]
            
            # Compute cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm_vec1 = np.linalg.norm(vec1)
            norm_vec2 = np.linalg.norm(vec2)
            similarity = dot_product / (norm_vec1 * norm_vec2)
            
            # Normalize cosine similarity from [-1,1] to [0,1]
            normalized_similarity = (similarity + 1) / 2
            
            # End timing the function execution
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Profiling output: Print the similarity score and time taken
            print(f"Cosine similarity between '{word1}' and '{word2}': {normalized_similarity:.4f}")
            print(f"Time taken: {elapsed_time:.6f} seconds")
            
            return normalized_similarity
        
        else:
            # Identify which word(s) are not in the vocabulary
            missing_words = [word for word in [word1, word2] if word not in model.key_to_index]
            print(f"Error: Word(s) not in vocabulary - {', '.join(missing_words)}")
            return None  # Indicates that similarity could not be computed
    
    except Exception as e:
        # Handle unexpected exceptions
        print(f"An error occurred while calculating cosine similarity: {e}")
        return None


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

def ngram_jaccard_similarity(word1, word2, n=2):
    ngrams1 = set([word1[i:i+n] for i in range(len(word1)-n+1)])
    ngrams2 = set([word2[i:i+n] for i in range(len(word2)-n+1)])
    intersection = ngrams1.intersection(ngrams2)
    union = ngrams1.union(ngrams2)
    return len(intersection) / len(union) if union else 0

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
