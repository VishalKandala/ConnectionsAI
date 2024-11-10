# src/model_loader.py

import os
import time
import logging
import gensim.downloader as api
from gensim.models import KeyedVectors
from config import EMBEDDINGS_PATH  # Import centralized path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelLoader:
    _vectors = None

    @classmethod
    def load_vectors(cls, model_path=None):
        if cls._vectors is None:
            if model_path is None:
                model_path = EMBEDDINGS_PATH  # Use centralized path
            
            if not os.path.isfile(model_path):
                logging.info(f"Model file not found at '{model_path}'. Initiating download of FastText vectors.")
                try:
                    logging.info("Downloading 'fasttext-wiki-news-subwords-300' model from Gensim's repository...")
                    fasttext_model = api.load('fasttext-wiki-news-subwords-300')  # This returns a KeyedVectors instance
                    logging.info("Download completed successfully.")
                except Exception as e:
                    logging.error(f"An error occurred while downloading the FastText model: {e}")
                    raise e

                # Ensure the 'embeddings' directory exists
                embeddings_dir = os.path.dirname(model_path)
                if not os.path.isdir(embeddings_dir):
                    os.makedirs(embeddings_dir)
                    logging.info(f"Created directory '{embeddings_dir}' for saving KeyedVectors.")

                # Save the KeyedVectors to the specified path
                try:
                    logging.info(f"Saving KeyedVectors to '{model_path}'...")
                    fasttext_model.save(model_path)
                    logging.info("KeyedVectors saved successfully.")
                except Exception as e:
                    logging.error(f"An error occurred while saving the KeyedVectors: {e}")
                    raise e

            try:
                logging.info(f"Loading word vectors from '{model_path}' with memory mapping...")
                start_time = time.time()
                cls._vectors = KeyedVectors.load(model_path, mmap='r')
                end_time = time.time()
                loading_time = end_time - start_time
                logging.info(f"Word vectors loaded successfully in {loading_time:.2f} seconds.")
            except Exception as e:
                logging.error(f"An error occurred while loading the model: {e}")
                raise e
        else:
            logging.info("Word vectors already loaded. Using cached version.")
        
        return cls._vectors
