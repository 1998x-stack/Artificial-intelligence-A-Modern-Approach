# 02_2.2.3_Searching_with_Nondeterministic_Actions

"""

Lecture: 2_Problem-solving/2.2_Beyond_Classical_Search
Content: 02_2.2.3_Searching_with_Nondeterministic_Actions

"""

import numpy as np
from typing import Callable, List, Tuple, Dict, Any

class NondeterministicSearch:
    """
    在非确定性操作中进行搜索的基类。
    """
    def __init__(self, initial_state: Any, goal_test: Callable[[Any], bool], actions: Callable[[Any], List[Callable[[Any], Any]]]):
        """
        初始化非确定性搜索。

        参数:
        - initial_state (Any): 初始状态。
        - goal_test (Callable[[Any], bool]): 测试目标状态的函数。
        - actions (Callable[[Any], List[Callable[[Any], Any]]]): 获取可用操作的函数。
        """
        self.initial_state = initial_state
        self.goal_test = goal_test
        self.actions = actions

    def results(self, state: Any, action: Callable[[Any], Any]) -> List[Any]:
        """
        获取给定状态和操作的所有可能结果。

        参数:
        - state (Any): 当前状态。
        - action (Callable[[Any], Any]): 施加于状态的操作。

        返回:
        - List[Any]: 所有可能的结果状态。
        """
        raise NotImplementedError("子类应该实现 results 方法。")

    def and_or_graph_search(self) -> Tuple[bool, List[Any]]:
        """
        执行AND-OR图搜索以找到解决方案。

        返回:
        - Tuple[bool, List[Any]]: 是否找到解决方案以及解决方案路径。
        """
        def or_search(state: Any, path: List[Any]) -> Tuple[bool, List[Any]]:
            if self.goal_test(state):
                return True, path
            if state in path:
                return False, []

            for action in self.actions(state):
                plan = [(action, and_search(result, path + [state])) for result in self.results(state, action)]
                if all(sub_plan[0] for _, sub_plan in plan):
                    return True, [action] + [sub_plan[1] for _, sub_plan in plan]
            return False, []

        def and_search(states: List[Any], path: List[Any]) -> Tuple[bool, List[Any]]:
            results = []
            for state in states:
                success, plan = or_search(state, path)
                if not success:
                    return False, []
                results.append(plan)
            return True, results

        return or_search(self.initial_state, [])

class VacuumWorld(NondeterministicSearch):
    """
    非确定性吸尘器世界。
    """
    def __init__(self, initial_state: Tuple[str, Tuple[bool, bool]], goal_test: Callable[[Any], bool]):
        """
        初始化非确定性吸尘器世界。

        参数:
        - initial_state (Tuple[str, Tuple[bool, bool]]): 初始状态。
        - goal_test (Callable[[Any], bool]): 测试目标状态的函数。
        """
        def actions(state: Tuple[str, Tuple[bool, bool]]) -> List[Callable[[Any], Any]]:
            return [self.suck, self.move_left, self.move_right]

        super().__init__(initial_state, goal_test, actions)

    def results(self, state: Tuple[str, Tuple[bool, bool]], action: Callable[[Any], Any]) -> List[Tuple[str, Tuple[bool, bool]]]:
        """
        获取给定状态和操作的所有可能结果。

        参数:
        - state (Tuple[str, Tuple[bool, bool]]): 当前状态。
        - action (Callable[[Any], Any]): 施加于状态的操作。

        返回:
        - List[Tuple[str, Tuple[bool, bool]]]: 所有可能的结果状态。
        """
        # 模拟非确定性操作的结果
        np.random.seed(42)  # 为了结果可重复
        if action == self.suck:
            if state[0] == 'A':
                return [('A', (False, state[1])), ('A', (False, not state[1]))]
            else:
                return [('B', (state[1], False)), ('B', (not state[1], False))]
        elif action == self.move_left:
            if state[0] == 'B':
                return [('A', state[1])]
            else:
                return [('A', state[1]), ('A', state[1])]
        elif action == self.move_right:
            if state[0] == 'A':
                return [('B', state[1])]
            else:
                return [('B', state[1]), ('B', state[1])]

    def suck(self, state: Tuple[str, Tuple[bool, bool]]) -> Tuple[str, Tuple[bool, bool]]:
        """
        吸尘操作。

        参数:
        - state (Tuple[str, Tuple[bool, bool]]): 当前状态。

        返回:
        - Tuple[str, Tuple[bool, bool]]: 吸尘后的新状态。
        """
        return state

    def move_left(self, state: Tuple[str, Tuple[bool, bool]]) -> Tuple[str, Tuple[bool, bool]]:
        """
        向左移动操作。

        参数:
        - state (Tuple[str, Tuple[bool, bool]]): 当前状态。

        返回:
        - Tuple[str, Tuple[bool, bool]]: 向左移动后的新状态。
        """
        return state

    def move_right(self, state: Tuple[str, Tuple[bool, bool]]) -> Tuple[str, Tuple[bool, bool]]:
        """
        向右移动操作。

        参数:
        - state (Tuple[str, Tuple[bool, bool]]): 当前状态。

        返回:
        - Tuple[str, Tuple[bool, bool]]: 向右移动后的新状态。
        """
        return state

# 示例用法：
if __name__ == "__main__":
    # 定义吸尘器世界的初始状态和目标测试函数
    initial_state = ('A', (True, True))  # 初始状态：位置A，两个格子都脏
    goal_test = lambda state: not state[1][0] and not state[1][1]

    # 创建非确定性吸尘器世界
    vacuum_world = VacuumWorld(initial_state, goal_test)

    # 执行AND-OR图搜索
    solution_found, solution_path = vacuum_world.and_or_graph_search()
    if solution_found:
        print("找到解决方案:", solution_path)
    else:
        print("未找到解决方案。")
