# 05_2.1.6_Heuristic_Functions

"""

Lecture: 2_Problem-solving/2.1_Solving_Problems_by_Searching
Content: 05_2.1.6_Heuristic_Functions

"""

import numpy as np
from typing import Tuple, List

class HeuristicFunctions:
    @staticmethod
    def manhattan_distance(start: Tuple[int, int], goal: Tuple[int, int]) -> int:
        """
        计算曼哈顿距离。

        Args:
        - start (Tuple[int, int]): 起始节点的坐标 (x, y)。
        - goal (Tuple[int, int]): 目标节点的坐标 (x, y)。

        Returns:
        - int: 曼哈顿距离。
        """
        return abs(start[0] - goal[0]) + abs(start[1] - goal[1])

    @staticmethod
    def euclidean_distance(start: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """
        计算欧几里得距离。

        Args:
        - start (Tuple[int, int]): 起始节点的坐标 (x, y)。
        - goal (Tuple[int, int]): 目标节点的坐标 (x, y)。

        Returns:
        - float: 欧几里得距离。
        """
        return np.sqrt((start[0] - goal[0]) ** 2 + (start[1] - goal[1]) ** 2)

    @staticmethod
    def misplaced_tiles(state: List[List[int]], goal: List[List[int]]) -> int:
        """
        计算拼图问题中的错位块数量。

        Args:
        - state (List[List[int]]): 当前拼图状态。
        - goal (List[List[int]]): 目标拼图状态。

        Returns:
        - int: 错位块数量。
        """
        state_arr = np.array(state)
        goal_arr = np.array(goal)
        return np.sum(state_arr != goal_arr) - 1  # 减去空格的比较

# 示例用法
start = (1, 2)
goal = (4, 6)

print(f"曼哈顿距离: {HeuristicFunctions.manhattan_distance(start, goal)}")
print(f"欧几里得距离: {HeuristicFunctions.euclidean_distance(start, goal):.2f}")

state = [
    [1, 2, 3],
    [4, 0, 5],
    [7, 8, 6]
]

goal_state = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 0]
]

print(f"错位块数量: {HeuristicFunctions.misplaced_tiles(state, goal_state)}")
