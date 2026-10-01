# 03_2.3.4_Imperfect_Real-Time_Decisions

"""

Lecture: 2_Problem-solving/2.3_Adversarial_Search
Content: 03_2.3.4_Imperfect_Real-Time_Decisions

"""

import numpy as np
from typing import Any, List, Tuple, Optional

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

    def get_reward(self) -> int:
        """
        获取当前状态的奖励值（仅对终局状态调用）。

        返回:
        - int: 当前状态的奖励值。
        """
        raise NotImplementedError("子类应实现 get_reward 方法。")


class MonteCarloTreeSearchNode:
    """
    蒙特卡罗树搜索节点类。
    """
    def __init__(self, state: GameState, parent: Optional['MonteCarloTreeSearchNode'] = None):
        """
        初始化蒙特卡罗树搜索节点。

        参数:
        - state (GameState): 当前节点的游戏状态。
        - parent (Optional[MonteCarloTreeSearchNode]): 父节点。
        """
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.reward = 0

    def is_fully_expanded(self) -> bool:
        """
        判断节点是否已经完全扩展。

        返回:
        - bool: 如果节点已经完全扩展，返回 True；否则返回 False。
        """
        return len(self.children) == len(self.state.get_legal_actions())

    def best_child(self, exploration_weight: float = 1.0) -> 'MonteCarloTreeSearchNode':
        """
        获取当前节点的最佳子节点。

        参数:
        - exploration_weight (float): 探索权重。

        返回:
        - MonteCarloTreeSearchNode: 最佳子节点。
        """
        choices_weights = [
            (child.reward / child.visits) + exploration_weight * np.sqrt((2 * np.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def expand(self) -> 'MonteCarloTreeSearchNode':
        """
        扩展当前节点的一个子节点。

        返回:
        - MonteCarloTreeSearchNode: 新扩展的子节点。
        """
        actions = self.state.get_legal_actions()
        for action in actions:
            if action not in [child.state for child in self.children]:
                new_state = self.state.apply_action(action)
                child_node = MonteCarloTreeSearchNode(new_state, self)
                self.children.append(child_node)
                return child_node
        raise Exception("无法扩展节点。")

    def rollout(self) -> int:
        """
        从当前节点进行随机模拟，直到到达终局状态。

        返回:
        - int: 模拟结果的奖励值。
        """
        current_state = self.state
        while not current_state.is_terminal():
            action = np.random.choice(current_state.get_legal_actions())
            current_state = current_state.apply_action(action)
        return current_state.get_reward()

    def backpropagate(self, reward: int):
        """
        进行回溯，更新当前节点及其所有祖先节点的访问次数和奖励值。

        参数:
        - reward (int): 模拟结果的奖励值。
        """
        self.visits += 1
        self.reward += reward
        if self.parent:
            self.parent.backpropagate(reward)


class MonteCarloTreeSearch:
    """
    蒙特卡罗树搜索算法类。
    """
    def __init__(self, state: GameState):
        """
        初始化蒙特卡罗树搜索算法。

        参数:
        - state (GameState): 初始游戏状态。
        """
        self.root = MonteCarloTreeSearchNode(state)

    def best_action(self, simulations_number: int) -> GameState:
        """
        执行给定次数的模拟，返回最佳动作。

        参数:
        - simulations_number (int): 模拟次数。

        返回:
        - GameState: 最佳动作后的新游戏状态。
        """
        for _ in range(simulations_number):
            node = self._tree_policy()
            reward = node.rollout()
            node.backpropagate(reward)
        return self.root.best_child(exploration_weight=0).state

    def _tree_policy(self) -> MonteCarloTreeSearchNode:
        """
        选择树中需要扩展的节点。

        返回:
        - MonteCarloTreeSearchNode: 需要扩展的节点。
        """
        node = self.root
        while not node.state.is_terminal():
            if not node.is_fully_expanded():
                return node.expand()
            else:
                node = node.best_child()
        return node

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
        for i in range(3):
            if abs(np.sum(self.board[i, :])) == 3 or abs(np.sum(self.board[:, i])) == 3:
                return True
        if abs(np.sum(self.board.diagonal())) == 3 or abs(np.sum(np.fliplr(self.board).diagonal())) == 3:
            return True
        if not np.any(self.board == 0):
            return True
        return False

    def get_reward(self) -> int:
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

    # 创建蒙特卡罗树搜索算法实例
    mcts = MonteCarloTreeSearch(initial_state)

    # 执行蒙特卡罗树搜索，找到最佳动作
    best_state = mcts.best_action(1000)
    print(f"最佳状态: \n{best_state.board}")
