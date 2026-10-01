# 01_2.1.2_Example_Problems

"""

Lecture: 2_Problem-solving/2.1_Solving_Problems_by_Searching
Content: 01_2.1.2_Example_Problems

"""

import numpy as np
from collections import deque
from typing import List, Tuple

class EightPuzzleSolver:
    def __init__(self, initial_state: List[List[int]], goal_state: List[List[int]]):
        """
        初始化八数码问题求解器。

        Args:
        - initial_state (List[List[int]]): 初始状态的 3x3 网格。
        - goal_state (List[List[int]]): 目标状态的 3x3 网格。
        """
        self.initial_state = np.array(initial_state)
        self.goal_state = np.array(goal_state)
        self.n = self.initial_state.shape[0]

    def is_solvable(self, state: np.ndarray) -> bool:
        """
        检查给定状态是否可解。

        Args:
        - state (np.ndarray): 需要检查的状态。

        Returns:
        - bool: 状态是否可解。
        """
        flattened = state.flatten()
        inv_count = sum(
            1 for i in range(len(flattened) - 1) for j in range(i + 1, len(flattened))
            if flattened[i] and flattened[j] and flattened[i] > flattened[j]
        )
        return inv_count % 2 == 0

    def bfs(self) -> List[np.ndarray]:
        """
        使用广度优先搜索（BFS）求解八数码问题。

        Returns:
        - List[np.ndarray]: 从初始状态到目标状态的状态序列。
        """
        if not self.is_solvable(self.initial_state):
            print("此初始状态不可解。")
            return []

        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        visited = set()
        queue = deque([(self.initial_state, [])])
        visited.add(tuple(self.initial_state.flatten()))

        while queue:
            state, path = queue.popleft()
            if np.array_equal(state, self.goal_state):
                return path + [state]

            zero_pos = tuple(np.argwhere(state == 0)[0])
            for move in moves:
                new_pos = (zero_pos[0] + move[0], zero_pos[1] + move[1])
                if 0 <= new_pos[0] < self.n and 0 <= new_pos[1] < self.n:
                    new_state = state.copy()
                    new_state[zero_pos], new_state[new_pos] = new_state[new_pos], new_state[zero_pos]
                    if tuple(new_state.flatten()) not in visited:
                        visited.add(tuple(new_state.flatten()))
                        queue.append((new_state, path + [state]))

        return []

# 示例用法
initial_state = [
    [1, 2, 3],
    [4, 0, 5],
    [7, 8, 6]
]

goal_state = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 0]
]

solver = EightPuzzleSolver(initial_state, goal_state)
solution_path = solver.bfs()

for step in solution_path:
    print(step)


import numpy as np
from itertools import permutations
from typing import List, Tuple

class TSPSolver:
    def __init__(self, distance_matrix: np.ndarray):
        """
        初始化旅行商问题求解器。

        Args:
        - distance_matrix (np.ndarray): 城市间的距离矩阵。
        """
        self.distance_matrix = distance_matrix
        self.n = distance_matrix.shape[0]

    def solve_brute_force(self) -> Tuple[float, List[int]]:
        """
        使用蛮力法求解旅行商问题。

        Returns:
        - Tuple[float, List[int]]: 最短路径的长度和城市访问顺序。
        """
        min_path_length = np.inf
        best_path = []

        for perm in permutations(range(self.n)):
            current_length = sum(
                self.distance_matrix[perm[i], perm[i + 1]] for i in range(self.n - 1)
            )
            current_length += self.distance_matrix[perm[-1], perm[0]]

            if current_length < min_path_length:
                min_path_length = current_length
                best_path = perm

        return min_path_length, list(best_path)

# 示例用法
distance_matrix = np.array([
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
])

tsp_solver = TSPSolver(distance_matrix)
min_length, best_path = tsp_solver.solve_brute_force()

print(f"最短路径长度: {min_length}")
print(f"最佳路径顺序: {best_path}")


import numpy as np
from typing import List, Tuple, Optional

class SudokuSolver:
    def __init__(self, board: List[List[int]]):
        """
        初始化数独求解器。

        Args:
        - board (List[List[int]]): 数独棋盘。
        """
        self.board = np.array(board)
        self.n = self.board.shape[0]

    def is_valid(self, num: int, pos: Tuple[int, int]) -> bool:
        """
        检查在给定位置填入数字是否有效。

        Args:
        - num (int): 要填入的数字。
        - pos (Tuple[int, int]): 填入的位置 (行, 列)。

        Returns:
        - bool: 填入数字是否有效。
        """
        row, col = pos

        if num in self.board[row, :]:
            return False

        if num in self.board[:, col]:
            return False

        box_x, box_y = row // 3, col // 3
        if num in self.board[box_x * 3:(box_x + 1) * 3, box_y * 3:(box_y + 1) * 3]:
            return False

        return True

    def find_empty(self) -> Optional[Tuple[int, int]]:
        """
        找到棋盘上的空位置。

        Returns:
        - Optional[Tuple[int, int]]: 空位置的坐标 (行, 列)，如果没有空位置则返回 None。
        """
        for i in range(self.n):
            for j in range(self.n):
                if self.board[i, j] == 0:
                    return i, j
        return None

    def solve(self) -> bool:
        """
        使用回溯算法求解数独问题。

        Returns:
        - bool: 是否找到解。
        """
        empty_pos = self.find_empty()
        if not empty_pos:
            return True

        row, col = empty_pos
        for num in range(1, 10):
            if self.is_valid(num, empty_pos):
                self.board[row, col] = num

                if self.solve():
                    return True

                self.board[row, col] = 0

        return False

# 示例用法
board = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

sudoku_solver = SudokuSolver(board)
if sudoku_solver.solve():
    print("数独解:")
    print(sudoku_solver.board)
else:
    print("无解")
