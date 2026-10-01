# 04_2.3.5_Stochastic_Games

"""

Lecture: 2_Problem-solving/2.3_Adversarial_Search
Content: 04_2.3.5_Stochastic_Games

"""

import numpy as np
from typing import List, Tuple, Dict, Callable, Any, Optional

class StochasticGameState:
    """
    随机博弈状态类，用于表示博弈中的一个状态。
    """
    def __init__(self, state: Any, player: int):
        """
        初始化游戏状态。

        参数:
        - state (Any): 当前的游戏状态，可以是任意类型。
        - player (int): 当前玩家（1 或 -1）。
        """
        self.state = state
        self.player = player

    def get_legal_actions(self) -> List[Any]:
        """
        获取当前状态下的所有合法动作。

        返回:
        - List[Any]: 合法动作的列表。
        """
        raise NotImplementedError("子类应实现 get_legal_actions 方法。")

    def apply_action(self, action: Any) -> 'StochasticGameState':
        """
        在当前状态下应用一个动作，返回新的游戏状态。

        参数:
        - action (Any): 要应用的动作。

        返回:
        - StochasticGameState: 应用动作后的新游戏状态。
        """
        raise NotImplementedError("子类应实现 apply_action 方法。")

    def is_terminal(self) -> bool:
        """
        判断当前状态是否为终局状态。

        返回:
        - bool: 如果是终局状态，返回 True；否则返回 False。
        """
        raise NotImplementedError("子类应实现 is_terminal 方法。")

    def get_reward(self) -> float:
        """
        获取当前状态的奖励值（仅对终局状态调用）。

        返回:
        - float: 当前状态的奖励值。
        """
        raise NotImplementedError("子类应实现 get_reward 方法。")

class MarkovDecisionProcess:
    """
    马尔可夫决策过程（MDP）类。
    """
    def __init__(self, states: List[Any], actions: List[Any], transition_probabilities: Dict[Tuple[Any, Any, Any], float], rewards: Dict[Tuple[Any, Any], float], discount_factor: float = 0.9):
        """
        初始化MDP。

        参数:
        - states (List[Any]): 状态列表。
        - actions (List[Any]): 动作列表。
        - transition_probabilities (Dict[Tuple[Any, Any, Any], float]): 状态转移概率，格式为{(s, a, s'): p}。
        - rewards (Dict[Tuple[Any, Any], float]): 奖励函数，格式为{(s, a): r}。
        - discount_factor (float): 折扣因子，默认为0.9。
        """
        self.states = states
        self.actions = actions
        self.transition_probabilities = transition_probabilities
        self.rewards = rewards
        self.discount_factor = discount_factor

    def value_iteration(self, epsilon: float = 1e-6) -> Dict[Any, float]:
        """
        价值迭代算法，用于求解MDP。

        参数:
        - epsilon (float): 收敛阈值。

        返回:
        - Dict[Any, float]: 最优状态值函数。
        """
        V = {s: 0 for s in self.states}
        while True:
            delta = 0
            for s in self.states:
                v = V[s]
                V[s] = max(sum(self.transition_probabilities.get((s, a, s_prime), 0) * (self.rewards.get((s, a), 0) + self.discount_factor * V[s_prime]) for s_prime in self.states) for a in self.actions)
                delta = max(delta, abs(v - V[s]))
            if delta < epsilon:
                break
        return V

    def extract_policy(self, V: Dict[Any, float]) -> Dict[Any, Any]:
        """
        根据最优状态值函数提取最优策略。

        参数:
        - V (Dict[Any, float]): 最优状态值函数。

        返回:
        - Dict[Any, Any]: 最优策略。
        """
        policy = {}
        for s in self.states:
            policy[s] = max(self.actions, key=lambda a: sum(self.transition_probabilities.get((s, a, s_prime), 0) * (self.rewards.get((s, a), 0) + self.discount_factor * V[s_prime]) for s_prime in self.states))
        return policy

# 示例：随机博弈中的井字棋状态类
class TicTacToeState(StochasticGameState):
    """
    随机博弈中的井字棋游戏状态类。
    """
    def __init__(self, board: np.ndarray, player: int):
        super().__init__(board, player)

    def get_legal_actions(self) -> List[Tuple[int, int]]:
        return [(i, j) for i in range(3) for j in range(3) if self.state[i, j] == 0]

    def apply_action(self, action: Tuple[int, int]) -> 'TicTacToeState':
        new_board = np.copy(self.state)
        new_board[action] = self.player
        return TicTacToeState(new_board, -self.player)

    def is_terminal(self) -> bool:
        for i in range(3):
            if abs(np.sum(self.state[i, :])) == 3 or abs(np.sum(self.state[:, i])) == 3:
                return True
        if abs(np.sum(self.state.diagonal())) == 3 or abs(np.sum(np.fliplr(self.state).diagonal())) == 3:
            return True
        if not np.any(self.state == 0):
            return True
        return False

    def get_reward(self) -> float:
        for i in range(3):
            if np.sum(self.state[i, :]) == 3 or np.sum(self.state[:, i]) == 3:
                return 1.0
            if np.sum(self.state[i, :]) == -3 or np.sum(self.state[:, i]) == -3:
                return -1.0
        if np.sum(self.state.diagonal()) == 3 or np.sum(np.fliplr(self.state).diagonal()) == 3:
            return 1.0
        if np.sum(self.state.diagonal()) == -3 or np.sum(np.fliplr(self.state).diagonal()) == -3:
            return -1.0
        return 0.0

# 示例用法：
if __name__ == "__main__":
    # 定义井字棋初始状态
    initial_board = np.zeros((3, 3), dtype=int)
    initial_state = TicTacToeState(initial_board, 1)

    # 定义状态、动作、转移概率和奖励函数
    states = [initial_state]
    actions = initial_state.get_legal_actions()
    transition_probabilities = {}  # 这里需要定义具体的转移概率
    rewards = {}  # 这里需要定义具体的奖励函数

    # 创建MDP实例
    mdp = MarkovDecisionProcess(states, actions, transition_probabilities, rewards)

    # 执行价值迭代
    V = mdp.value_iteration()
    print("最优状态值函数:", V)

    # 提取最优策略
    policy = mdp.extract_policy(V)
    print("最优策略:", policy)
