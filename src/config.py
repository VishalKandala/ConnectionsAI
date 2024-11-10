# src/config.py

"""
config.py

This module centralizes configuration settings for the project, including file paths and other constants.
"""

import os

# Determine the absolute path to the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Define paths relative to the project root
EMBEDDINGS_PATH = os.path.join(PROJECT_ROOT, 'embeddings', 'fasttext_vectors.kv')
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'sample_data')
