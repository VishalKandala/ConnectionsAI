# src/model_loader.py

###############################################################################
#                                                                             #
#                               Model Loader                                  #
#                                                                             #
#      A module to load the pre-trained FastText model using gensim.          #
#                                                                             #
###############################################################################

# Import necessary libraries
import gensim.downloader as api  # For downloading pre-trained models

def load_model():
    """
    Load the pre-trained FastText model using gensim.

    Returns:
    - model: The loaded FastText model.
    """
    print("Loading the FastText model...")
    model = api.load('fasttext-wiki-news-subwords-300')  # This may take some time
    print("Model loaded successfully.")
    return model
