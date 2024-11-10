
# Connections AI

A Flask-based application that groups words into categories based on semantic, lexical, and spelling similarities using natural language processing techniques and string metrics.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Usage](#usage)
- [Modules Description](#modules-description)
- [Dependencies](#dependencies)
- [Notes](#notes)
- [License](#license)

---

## Overview

**Connections AI** is designed to receive a list of words and group them into categories based on their similarities. The application leverages pre-trained FastText word embeddings via the `gensim` library to compute semantic similarities, as well as Jaccard similarity and Levenshtein distance for lexical and spelling comparisons.

This application is structured to work with a provided Flask server (`app.py`) that calls a `model` function defined in `src/Model.py`. The goal is to process the input data and return a valid guess and an indication of whether to end the turn.

---

## Project Structure

```
your_project/
├── app.py
├── src/
│   ├── __init__.py
│   ├── Model.py
│   ├── model_loader.py
│   ├── similarity_metrics.py
│   └── connections_model.py
├── tests/
│   ├── evaluator.py
│   ├── sample_data.json
├── requirements.txt
├── README.md

```

- **`app.py`**: The Flask application that handles incoming requests and invokes the `model` function.
- **`src/`**: Contains all the source code modules.
  - **`Model.py`**: Contains the `model` function that integrates with the Flask app.
  - **`model_loader.py`**: Loads the pre-trained FastText model using `gensim`.
  - **`similarity_metrics.py`**: Functions for calculating cosine similarity, Jaccard similarity, and Levenshtein distance.
  - **`connections_model.py`**: Contains the main logic for grouping words.
  - **`__init__.py`**: Makes `src` a Python package.
- **`tests/`**: Contains all the source code modules.
  - **`evaluator.py`**: Has a script which can be used to evaluate the performance of the entire project
  - **`sample_data.json`**: Data to test performance against.
- **`requirements.txt`**: Lists all the Python dependencies.
- **`README.md`**: This file, providing an overview and instructions.

---

## Installation

### Prerequisites

- Python 3.10 or higher
- `pip` package manager

### Steps 
1. **Clone the Repository**

   ```bash
   git clone https://github.com/VishalKandala/ConnectionsAI.git
   cd ConnectionsAI
   ```

3. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   This will install all the necessary packages listed in `requirements.txt`.

---

## Running the Application

1. **Start the Flask Server**

   ```bash
   python app.py
   ```

   The server will start and listen on port `5000`.

 ** Open a new terminal **

2. **Send a POST Request**
 
   You can use `curl`, `Postman`, or any HTTP client to send a POST request to the server.

   **Example using `curl`:**

   ```bash
   curl -X POST http://localhost:5000/ -H "Content-Type: application/json" -d '{
       "words": ["apple", "orange", "grape", "banana", "cat", "dog", "lion", "tiger", "night", "knight", "day", "sun", "moon", "star", "pen", "pencil"],
       "strikes": 0,
       "isOneAway": false,
       "correctGroups": [],
       "previousGuesses": [],
       "error": "0"
   }'
   ```

3. **Expected Response**

   The server will process the request and return a JSON response containing the `guess` and `endTurn` fields.

   ```json
   {
       "guess": ["apple", "orange", "grape", "banana"],
       "endTurn": false
   }
   ```

---

## Usage

- **Input Parameters**:
  - `words`: A list of 16 shuffled words to be grouped.
  - `strikes`: An integer representing the number of strikes.
  - `isOneAway`: A boolean indicating if the previous guess was one word away from the correct answer.
  - `correctGroups`: A list of groups that have been correctly guessed in previous turns.
  - `previousGuesses`: A list of previous guesses made.
  - `error`: A string containing an error message (if any).

- **Output**:
  - `guess`: A list of 4 words as the next guess.
  - `endTurn`: A boolean indicating whether to end the puzzle.

---

## Modules Description

### 1. `app.py`

- The main Flask application.
- Receives POST requests and extracts input data.
- Calls the `model` function from `src/Model.py`.
- Returns the guess and endTurn decision as a JSON response.

### 2. `src/Model.py`

- Contains the `model` function required by the Flask app.
- Loads the FastText model upon import (to avoid reloading it every time).
- Processes the input words and determines the next guess.
- Ensures that words already used in `correctGroups` or `previousGuesses` are not guessed again.

### 3. `src/model_loader.py`

- Loads the pre-trained FastText model using `gensim`.
- The `load_model` function returns the loaded model.
- Prints status messages during loading.

### 4. `src/similarity_metrics.py`

- Contains functions to calculate:
  - **Cosine Similarity** (`calculate_cosine_similarity`): Measures semantic similarity using word embeddings.
  - **Jaccard Similarity** (`calculate_jaccard_similarity`): Measures lexical similarity based on shared characters.
  - **Levenshtein Distance** (`calculate_levenshtein_distance`): Measures spelling similarity based on edit distance.

### 5. `src/connections_model.py`

- Implements the `connections_model` function to group words.
- Groups words based on:
  - Semantic similarity (cosine similarity).
  - Lexical similarity (Jaccard similarity).
  - Spelling similarity (Levenshtein distance).
- Ensures that each group contains exactly four words.
- Returns a dictionary of grouped words.

---

## Dependencies

Listed in `requirements.txt`:

- **flask**
- **gensim**
- **numpy**
- **python-Levenshtein**

**Installation Command:**

```bash
pip install -r requirements.txt
```

---

## Notes

- **Model Loading**: The FastText model (`fasttext-wiki-news-subwords-300`) is approximately 1GB in size. The first time you run the application, it may take some time to download and load the model.
- **Adjusting Similarity Thresholds**: The thresholds for cosine similarity, Jaccard similarity, and Levenshtein distance are set in `connections_model.py`. Adjust them as needed for your specific use case.
- **Out-of-Vocabulary Words**: If a word is not found in the model's vocabulary, the cosine similarity function assigns a minimal similarity score to avoid errors.
- **Performance Considerations**: Loading the model once and reusing it improves performance. Ensure that the model is loaded at the module level and not within a function that's called repeatedly.
- **Extensibility**: The modular design allows for easy extension. You can add new similarity metrics or modify existing ones without affecting other parts of the code.
- **Testing**: You can write unit tests for individual modules to ensure correctness.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
