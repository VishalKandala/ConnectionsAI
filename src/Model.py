# src/Model.py

###############################################################################
#                                                                             #
#                                Model Module                                 #
#                                                                             #
#      Contains the model function that integrates with the Flask app.        #
#                                                                             #
###############################################################################

# Import necessary modules
from .model_loader import load_model  # Function to load the FastText model
from .connections_model import connections_model  # Function to group words

# Load the pre-trained FastText model once when the module is imported
model_instance = load_model()

def model(words, strikes, isOneAway, correctGroups, previousGuesses, error):
    """
    _______________________________________________________
    Parameters:
    words - 1D Array with 16 shuffled words
    strikes - Integer with number of strikes
    isOneAway - Boolean if your previous guess is one word away from the correct answer
    correctGroups - 2D Array with groups previously guessed correctly
    previousGuesses - 2D Array with previous guesses
    error - String with error message (0 if no error)

    Returns:
    guess - 1D Array with 4 words
    endTurn - Boolean if you want to end the puzzle
    _______________________________________________________
    """
    # Get the groups using connections_model
    groups = connections_model(words, model_instance)

    # Flatten correctGroups and previousGuesses to get words already used
    used_words = set()
    for group in correctGroups + previousGuesses:
        used_words.update(group)

    # Find a group that hasn't been guessed yet
    guess = None
    for group_name, group_words in groups.items():
        if not any(word in used_words for word in group_words):
            guess = group_words
            break

    if guess:
        endTurn = False  # Decide whether to end turn or not
    else:
        # No new groups found, decide whether to end turn
        guess = []  # No guess
        endTurn = True  # No more guesses available

    return guess, endTurn
