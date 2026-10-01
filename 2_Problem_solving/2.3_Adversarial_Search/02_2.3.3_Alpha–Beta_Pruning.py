# 02_2.3.3_Alpha–Beta_Pruning

"""

Lecture: 2_Problem-solving/2.3_Adversarial_Search
Content: 02_2.3.3_Alpha–Beta_Pruning

"""

import numpy as np
from typing import List, Tuple, Callable, Any

class GameState:
    """
    游戏状态类，用于表示博弈中的一个状态。
    """
    def __init__(self, board: np.ndarray, player: int):
        """
        初始化游戏状态。

        参数:
        - board (np.ndarray): 当前的棋盘状态。
        - player (int): 当前玩家（1或-1）。
        """
        self.board = board
        self.player = player

    def get_legal_actions(self) -> List[Tuple[int, int]]:
        """
        获取当前状态下的所有合法动作。

        返回:
        - List[Tuple[int, int]]: 合法动作的列表。
        """
        raise NotImplementedError("子类应实现 get_legal_actions 方法。")

    def apply_action(self, action: Tuple[int, int]) -> 'GameState':
        """
        在当前状态下应用一个动作，返回新的游戏状态。

        参数:
        - action (Tuple[int, int]): 要应用的动作。

        返回:
        - GameState: 应用动作后的新游戏状态。
        """
        raise NotImplementedError("子类应实现 apply_action 方法。")

    def is_terminal(self) -> bool:
        """
        判断当前状态是否为终局状态。

        返回:
        - bool: 如果是终局状态，返回 True；否则返回 False。
        """
        raise NotImplementedError("子类应实现 is_terminal 方法。")

    def get_utility(self) -> int:
        """
        获取当前状态的效用值（仅对终局状态调用）。

        返回:
        - int: 当前状态的效用值。
        """
        raise NotImplementedError("子类应实现 get_utility 方法。")


class AlphaBetaPruning:
    """
    Alpha-Beta剪枝算法类。
    """
    def __init__(self, initial_state: GameState):
        """
        初始化Alpha-Beta剪枝算法。

        参数:
        - initial_state (GameState): 初始游戏状态。
        """
        self.initial_state = initial_state

    def alpha_beta_search(self) -> Tuple[Any, int]:
        """
        执行Alpha-Beta剪枝搜索，返回最佳动作及其效用值。

        返回:
        - Tuple[Any, int]: 最佳动作及其效用值。
        """
        def max_value(state: GameState, alpha: float, beta: float) -> int:
            if state.is_terminal():
                return state.get_utility()
            value = -np.inf
            for action in state.get_legal_actions():
                value = max(value, min_value(state.apply_action(action), alpha, beta))
                if value >= beta:
                    return value
                alpha = max(alpha, value)
            return value

        def min_value(state: GameState, alpha: float, beta: float) -> int:
            if state.is_terminal():
                return state.get_utility()
            value = np.inf
            for action in state.get_legal_actions():
                value = min(value, max_value(state.apply_action(action), alpha, beta))
                if value <= alpha:
                    return value
                beta = min(beta, value)
            return value

        best_action = None
        best_value = -np.inf
        for action in self.initial_state.get_legal_actions():
            value = min_value(self.initial_state.apply_action(action), -np.inf, np.inf)
            if value > best_value:
                best_value = value
                best_action = action
        return best_action, best_value

# 示例游戏状态类（以井字棋为例）
class TicTacToeState(GameState):
    """
    井字棋游戏状态类。
    """
    def get_legal_actions(self) -> List[Tuple[int, int]]:
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]

    def apply_action(self, action: Tuple[int, int]) -> 'TicTacToeState':
        new_board = np.copy(self.board)
        new_board[action] = self.player
        return TicTacToeState(new_board, -self.player)

    def is_terminal(self) -> bool:
        # 检查行、列和对角线是否有相同的标记
        for i in range(3):
            if abs(np.sum(self.board[i, :])) == 3 or abs(np.sum(self.board[:, i])) == 3:
                return True
        if abs(np.sum(self.board.diagonal())) == 3 or abs(np.sum(np.fliplr(self.board).diagonal())) == 3:
            return True
        # 检查是否还有空位
        if not np.any(self.board == 0):
            return True
        return False

    def get_utility(self) -> int:
        for i in range(3):
            if np.sum(self.board[i, :]) == 3 or np.sum(self.board[:, i]) == 3:
                return 1
            if np.sum(self.board[i, :]) == -3 or np.sum(self.board[:, i]) == -3:
                return -1
        if np.sum(self.board.diagonal()) == 3 or np.sum(np.fliplr(self.board).diagonal()) == 3:
            return 1
        if np.sum(self.board.diagonal()) == -3 or np.sum(np.fliplr(self.board).diagonal()) == -3:
            return -1
        return 0

# 示例用法：
if __name__ == "__main__":
    # 初始化井字棋初始状态
    initial_board = np.zeros((3, 3), dtype=int)
    initial_state = TicTacToeState(initial_board, 1)

    # 创建Alpha-Beta剪枝算法实例
    alpha_beta = AlphaBetaPruning(initial_state)

    # 执行Alpha-Beta剪枝搜索
    best_action, best_value = alpha_beta.alpha_beta_search()
    print(f"最佳动作: {best_action}, 最优值: {best_value}")
