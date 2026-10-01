# 01_2.3.2_Optimal_Decisions_in_Games

"""

Lecture: 2_Problem-solving/2.3_Adversarial_Search
Content: 01_2.3.2_Optimal_Decisions_in_Games

"""

import numpy as np
from typing import Tuple

class Minimax:
    def __init__(self, evaluation_function):
        """
        Initialize the Minimax algorithm with an evaluation function.

        Parameters:
        - evaluation_function: Function to evaluate the utility of a game state.
        """
        self.eval_func = evaluation_function

    def minimax_decision(self, state: np.ndarray) -> Tuple[int, float]:
        """
        Perform Minimax decision to find the optimal action and its utility.

        Parameters:
        - state: Current game state represented as a numpy array.

        Returns:
        - Tuple containing optimal action (index) and its utility (float).
        """
        # Ensure the state is represented correctly (example)
        # Here, `state` is assumed to be a numpy array or matrix representing the game state.

        # Determine all possible actions (example)
        actions = self.get_possible_actions(state)

        best_utility = -np.inf
        best_action = None

        # For each action, compute the utility using Minimax
        for action in actions:
            # Apply action to the state (hypothetically)
            new_state = self.apply_action(state, action)
            
            # Compute utility using Minimax
            utility = self.min_value(new_state)

            # Update best action if this action provides a higher utility
            if utility > best_utility:
                best_utility = utility
                best_action = action

        return best_action, best_utility

    def min_value(self, state: np.ndarray) -> float:
        """
        Minimize the utility value for the opponent's turn.

        Parameters:
        - state: Current game state represented as a numpy array.

        Returns:
        - Minimum utility value achievable for the opponent.
        """
        # Check if the game is over or evaluate the state if terminal
        if self.is_terminal(state):
            return self.eval_func(state)

        min_utility = np.inf

        # Determine all possible actions (example)
        actions = self.get_possible_actions(state)

        # For each action, compute the utility using Minimax
        for action in actions:
            # Apply action to the state (hypothetically)
            new_state = self.apply_action(state, action)
            
            # Compute utility using Max-value function
            utility = self.max_value(new_state)

            # Update minimum utility if this action provides a lower utility
            if utility < min_utility:
                min_utility = utility

        return min_utility

    def max_value(self, state: np.ndarray) -> float:
        """
        Maximize the utility value for the player's turn.

        Parameters:
        - state: Current game state represented as a numpy array.

        Returns:
        - Maximum utility value achievable for the player.
        """
        # Check if the game is over or evaluate the state if terminal
        if self.is_terminal(state):
            return self.eval_func(state)

        max_utility = -np.inf

        # Determine all possible actions (example)
        actions = self.get_possible_actions(state)

        # For each action, compute the utility using Minimax
        for action in actions:
            # Apply action to the state (hypothetically)
            new_state = self.apply_action(state, action)
            
            # Compute utility using Min-value function
            utility = self.min_value(new_state)

            # Update maximum utility if this action provides a higher utility
            if utility > max_utility:
                max_utility = utility

        return max_utility

    def apply_action(self, state: np.ndarray, action) -> np.ndarray:
        """
        Apply the action to the current state and return the new state.

        Parameters:
        - state: Current game state represented as a numpy array.
        - action: Action to be applied to the state.

        Returns:
        - New game state after applying the action.
        """
        # Example implementation: Apply action to the state
        new_state = state.copy()
        new_state[action] = 1  # Example action application
        
        return new_state

    def is_terminal(self, state: np.ndarray) -> bool:
        """
        Check if the current state is a terminal state (end of game).

        Parameters:
        - state: Current game state represented as a numpy array.

        Returns:
        - True if the state is terminal, False otherwise.
        """
        # Example implementation: Check if the state is terminal
        # Example condition: game is over when no more actions are possible
        return len(self.get_possible_actions(state)) == 0

    def get_possible_actions(self, state: np.ndarray):
        """
        Get all possible actions that can be taken from the current state.

        Parameters:
        - state: Current game state represented as a numpy array.

        Returns:
        - List of possible actions.
        """
        # Example implementation: Get possible actions (e.g., indices of empty cells)
        return np.where(state == 0)[0]

    def evaluation_function(self, state: np.ndarray) -> float:
        """
        Evaluate the utility of the current game state.

        Parameters:
        - state: Current game state represented as a numpy array.

        Returns:
        - Utility value indicating the goodness of the state for the current player.
        """
        # Example implementation: Simple evaluation function (e.g., count number of filled rows/columns)
        # Replace with a more sophisticated evaluation function depending on the game
        return np.sum(state)  # Example: Count number of filled cells (assuming 1 represents filled)

# Example usage:
if __name__ == "__main__":
    # Initialize Minimax with an evaluation function (example)
    minimax = Minimax(evaluation_function=minimax.evaluation_function)

    # Example game state (numpy array)
    game_state = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, 0]
    ])

    # Perform Minimax decision
    best_action, best_utility = minimax.minimax_decision(game_state)
    print(f"Best Action: {best_action}, Best Utility: {best_utility}")
