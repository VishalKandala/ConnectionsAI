import random
import numpy as np
import requests
import json
import os

def evalFunction():
    # Load puzzles
    puzzles = load_puzzles()
    totalPoints = 0

    z = 1
    invalidGuesses = 0

    for puzzle in puzzles:
        shuffledPuzzle = shufflePuzzles(puzzle)

        # Initialize all variables
        strikes = 0
        correctGroups = []
        previousGuesses = []
        error = 0
        isOneAway = False

        while strikes < 4 and len(correctGroups) < 4 and invalidGuesses < 7:
            data = {
                "words": shuffledPuzzle,
                "strikes": strikes,
                "isOneAway": isOneAway,
                "correctGroups": correctGroups,
                "previousGuesses": previousGuesses,
                "error": error
            }
            headers = {'Content-Type': 'application/json'}

            # Send POST request to the Flask app
            r = requests.post("http://127.0.0.1:5000", json=data, headers=headers)
            response = r.json()
            participantGuess = response['guess']
            print("Participant guess: ", participantGuess)
            endTurn = response['endTurn']

            sortedGuess = sorted(participantGuess)
            if any(sortedGuess == sorted(x) for x in previousGuesses):
                error = "You have already guessed this combination."
                invalidGuesses += 1
                continue
            else:
                error = 0
                previousGuesses.append(participantGuess)

            if len(participantGuess) != 4:
                error = "Please enter 4 words."
                invalidGuesses += 1
                continue

            if endTurn:
                break

            correctlyGuessed = False
            for group in puzzle:
                sortedGroup = sorted(group)
                if sortedGroup == sortedGuess:
                    correctlyGuessed = True
                    correctGroups.append(group)
                    break
                else:
                    set1, set2 = set(group), set(participantGuess)
                    if len(set1.symmetric_difference(set2)) == 2:
                        isOneAway = True
                        print("One away")
                        break
                    else:
                        isOneAway = False

            if not correctlyGuessed:
                strikes += 1

        # Calculate points
        points = 0

        for i in range(len(correctGroups)):
            # Group multiplier based on the group number
            if i + 1 == 1:
                groupMult = 1
            elif i + 1 == 2:
                groupMult = 2
            elif i + 1 == 3:
                groupMult = 3
            elif i + 1 == 4:
                groupMult = 3

            # Strike multiplier based on the number of strikes
            if strikes == 0:
                strikeMult = 1
            elif strikes == 1:
                strikeMult = 0.9
            elif strikes == 2:
                strikeMult = 0.75
            elif strikes == 3:
                strikeMult = 0.5
            elif strikes == 4:
                strikeMult = 0.25

            points += groupMult * strikeMult
            print(f"Points scored by model on puzzle {z}, group {i+1}: {groupMult * strikeMult}")
        totalPoints += points
        z += 1

    # Store total points
    print("Total points scored by model: ", totalPoints)

def load_puzzles():
    import os
    # Get the directory of the current script (tests directory)
    script_dir = os.path.dirname(__file__)
    # Build the path to sample_data.json
    data_file_path = os.path.join(script_dir, 'sample_data.json')

    with open(data_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

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
