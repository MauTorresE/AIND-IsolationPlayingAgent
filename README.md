
# Build a Game-playing Agent

## Synopsis

In this project, students will develop an adversarial search agent to play the game "Isolation".  Students only need to modify code in the `game_agent.py`, however, code is included for example player and evaluation functions for you to review and test against in the other files.

Isolation is a deterministic, two-player game of perfect information in which the players alternate turns moving a single piece from one cell to another on a board.  Whenever either player occupies a cell, that cell becomes blocked for the remainder of the game.  The first player with no remaining legal moves loses, and the opponent is declared the winner.

This project uses a version of Isolation where each agent is restricted to L-shaped movements (like a knight in chess) on a rectangular grid (like a chess or checkerboard).  The agents can move to any open cell on the board that is 2-rows and 1-column or 2-columns and 1-row away from their current position on the board. Movements are blocked at the edges of the board (the board does not wrap around), however, the player can "jump" blocked or occupied spaces (just like a knight in chess).

Additionally, agents will have a fixed time limit each turn to search for the best move and respond.  If the time limit expires during a player's turn, that player forfeits the match, and the opponent wins.

These rules are implemented in the `isolation.Board` class provided in the repository. 


## Instructions

Implement the following four functions in `game_agent.py`:

- `CustomPlayer.minimax()`: implement minimax search
- `CustomPlayer.alphabeta()`: implement minimax search with alpha-beta pruning
- `CustomPlayer.get_move()`: implement fixed-depth and iterative deepening search
- `custom_score()`: implement your own position evaluation heuristic

You may write or modify code within each file (as long as you maintain compatibility with the function signatures provided) and you may add other classes, functions, etc., as needed, but it is not required.  


### Tournament

The `tournament.py` script is used to evaluate the effectiveness of your custom_score heuristic.  The script measures relative performance of your agent (called "Student") in a round-robin tournament against several other pre-defined agents.  The Student agent uses time-limited Iterative Deepening and the custom_score heuristic you wrote.

The performance of time-limited iterative deepening search is hardware dependent (faster hardware is expected to search deeper than slower hardware in the same amount of time).  The script controls for these effects by also measuring the baseline performance of an agent called "ID_Improved" that uess Iterative Deepening and the improved_score heuristic from `sample_players.py`.  Your goal is to develop a heuristic such that Student outperforms ID_Improved.

The tournament opponents are listed below. (See also: sample heuristics and players defined in sample_players.py)

- Random: An agent that randomly chooses a move each turn.
- MM_Null: CustomPlayer agent using fixed-depth minimax search and the null_score heuristic
- MM_Open: CustomPlayer agent using fixed-depth minimax search and the open_move_score heuristic
- MM_Improved: CustomPlayer agent using fixed-depth minimax search and the improved_score heuristic
- AB_Null: CustomPlayer agent using fixed-depth alpha-beta search and the null_score heuristic
- AB_Open: CustomPlayer agent using fixed-depth alpha-beta search and the open_move_score heuristic
- AB_Improved: CustomPlayer agent using fixed-depth alpha-beta search and the improved_score heuristic
# AIND-IsolationPlayingAgent
