import sys
import os

# Adjust the module search path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import random
import numpy as np
import json

# Import the model function
from src.Model import model

GROUP_MULTIPLIERS = {1: 1, 2: 2, 3: 3, 4: 3}
STRIKE_MULTIPLIERS = {0: 1, 1: 0.9, 2: 0.75, 3: 0.5, 4: 0.25}

def evalFunction():
    # Load puzzles
    puzzles = load_puzzles()
    totalPoints = 0
    puzzle_counter = 1  # Keep track of the number of puzzles played

    while True:
        # Randomly select a puzzle
        puzzle = random.choice(puzzles)
        shuffledPuzzle = shufflePuzzles(puzzle)

        # Initialize all variables
        strikes = 0
        correctGroups = []
        previousGuesses = []
        error = ""
        invalidGuesses = 0
        isOneAway = False
        endTurn = False  # Initialize endTurn

        # Print the correct groups for debugging
        print(f"Correct groups for Puzzle {puzzle_counter}:")
        for i, group in enumerate(puzzle):
            print(f"  Group {i+1}: {group}")

        print(f"\nStarting Puzzle {puzzle_counter}")
        print("-----------------------------")
        print(f"Shuffled words: {shuffledPuzzle}\n")

        while strikes < 4 and len(correctGroups) < 4 and invalidGuesses < 7 and not endTurn:
            isOneAway = False  # Reset at the beginning of each turn

            # Print data passed to model (for debugging)
            print("Data passed to model:")
            print(f"  shuffledPuzzle: {shuffledPuzzle}")
            print(f"  strikes: {strikes}")
            print(f"  isOneAway: {isOneAway}")
            print(f"  correctGroups: {correctGroups}")
            print(f"  previousGuesses: {previousGuesses}")
            print(f"  error: {error}")

            # Call the model function
            try:
                participantGuess, endTurn = model(
                    words=shuffledPuzzle,
                    strikes=strikes,
                    isOneAway=isOneAway,
                    correctGroups=correctGroups,
                    previousGuesses=previousGuesses,
                    error=error
                )
            except Exception as e:
                print(f"An error occurred while calling the model: {e}")
                error = "Model encountered an error."
                invalidGuesses += 1
                continue

            print(f"Participant guess: {participantGuess}")

            if endTurn:
                print("Model decided to end the turn.")
                break  # Exit the while loop

            # Validate participantGuess
            if not isinstance(participantGuess, list):
                error = "Model returned an invalid guess."
                invalidGuesses += 1
                continue
            if len(participantGuess) != 4:
                error = "Please enter 4 words."
                invalidGuesses += 1
                continue

            sortedGuess = sorted([word.upper() for word in participantGuess])
            if any(sortedGuess == sorted([word.upper() for word in x]) for x in previousGuesses):
                error = "You have already guessed this combination."
                invalidGuesses += 1
                continue
            else:
                error = ""
                previousGuesses.append(participantGuess)

            correctlyGuessed = False
            for group in puzzle:
                sortedGroup = sorted([word.upper() for word in group])
                if sortedGroup == sortedGuess:
                    correctlyGuessed = True
                    correctGroups.append(group)
                    print("Correct group guessed!")
                    break
                else:
                    set1, set2 = set(sortedGroup), set(sortedGuess)
                    if len(set1.symmetric_difference(set2)) == 2:
                        isOneAway = True
                        print("One away")
                        break
                    else:
                        isOneAway = False

            if not correctlyGuessed:
                strikes += 1
                print(f"Incorrect guess. Strikes: {strikes}")

            # Wait before proceeding to the next attempt
            input("Press Enter to proceed to the next attempt...")

        # Calculate points
        points = 0

        for i, group in enumerate(correctGroups):
            groupMult = GROUP_MULTIPLIERS.get(i + 1, 1)
            strikeMult = STRIKE_MULTIPLIERS.get(strikes, 0.25)
            groupPoints = groupMult * strikeMult
            points += groupPoints

            # Output the group number, words, and points scored
            print(f"Puzzle {puzzle_counter}, Group {i + 1}:")
            print(f"  Words: {group}")
            print(f"  Points scored: {groupPoints}")
            print("-----------------------------")

        print(f"Total points scored in Puzzle {puzzle_counter}: {points}")
        print("=============================\n")

        totalPoints += points
        puzzle_counter += 1

        # Prompt the user to decide whether to continue
        user_input = input("Press Enter to play another puzzle or type 'exit' to quit: ").strip().lower()
        if user_input in ['exit', 'quit']:
            break  # Exit the while loop
        else:
            print("\nStarting a new puzzle...\n")

    # Final total points
    print(f"Total points scored by model after {puzzle_counter - 1} puzzles: {totalPoints}")

def load_puzzles():
    # Get the directory of the current script (tests directory)
    script_dir = os.path.dirname(__file__)
    # Build the path to sample_data.json
    data_file_path = os.path.join(script_dir, 'sample_data.json')

    try:
        with open(data_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"Data file not found at {data_file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return []

    # Create a 3D array (X puzzles, 4 rows, 4 words)
    puzzles_3d = []

    for puzzle in data:
        # Extract only the words part for each puzzle
        puzzle_words = [entry["words"] for entry in puzzle]
        puzzles_3d.append(puzzle_words)

    return puzzles_3d

def shufflePuzzles(puzzle):
    # Flatten the puzzle into a 1D list
    flattened_puzzle = np.array(puzzle).flatten()
    # Shuffle the words
    np.random.shuffle(flattened_puzzle)
    # Convert NumPy array to a Python list
    shuffled_puzzle = flattened_puzzle.tolist()
    return shuffled_puzzle

if __name__ == "__main__":
    evalFunction()
