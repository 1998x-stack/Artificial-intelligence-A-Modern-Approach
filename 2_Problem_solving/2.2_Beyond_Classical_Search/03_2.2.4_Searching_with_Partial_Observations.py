# 03_2.2.4_Searching_with_Partial_Observations

"""

Lecture: 2_Problem-solving/2.2_Beyond_Classical_Search
Content: 03_2.2.4_Searching_with_Partial_Observations

"""

import numpy as np
from typing import List, Tuple, Callable, Any, Dict

class BeliefState:
    """
    信念状态类，用于表示代理人在部分观测环境下的信念状态。
    """
    def __init__(self, states: List[Any]):
        """
        初始化信念状态。

        参数:
        - states (List[Any]): 可能的物理状态列表。
        """
        self.states = states

    def update(self, percept: Any, percept_fn: Callable[[Any], Any]) -> 'BeliefState':
        """
        根据新的感知信息更新信念状态。

        参数:
        - percept (Any): 当前感知信息。
        - percept_fn (Callable[[Any], Any]): 感知函数，给定物理状态返回感知信息。

        返回:
        - BeliefState: 更新后的信念状态。
        """
        new_states = [state for state in self.states if percept_fn(state) == percept]
        return BeliefState(new_states)


class PartialObservationsSearch:
    """
    在部分观测环境下进行搜索的基类。
    """
    def __init__(self, initial_belief: BeliefState, goal_test: Callable[[Any], bool], actions_fn: Callable[[Any], List[Callable[[Any], Any]]], percept_fn: Callable[[Any], Any]):
        """
        初始化部分观测环境下的搜索。

        参数:
        - initial_belief (BeliefState): 初始信念状态。
        - goal_test (Callable[[Any], bool]): 测试目标状态的函数。
        - actions_fn (Callable[[Any], List[Callable[[Any], Any]]]): 获取可用操作的函数。
        - percept_fn (Callable[[Any], Any]): 感知函数，给定物理状态返回感知信息。
        """
        self.initial_belief = initial_belief
        self.goal_test = goal_test
        self.actions_fn = actions_fn
        self.percept_fn = percept_fn

    def predict(self, belief: BeliefState, action: Callable[[Any], Any]) -> BeliefState:
        """
        预测执行某个动作后的信念状态。

        参数:
        - belief (BeliefState): 当前信念状态。
        - action (Callable[[Any], Any]): 动作函数。

        返回:
        - BeliefState: 预测的信念状态。
        """
        new_states = [action(state) for state in belief.states]
        return BeliefState(new_states)

    def and_or_search(self) -> Tuple[bool, List[Any]]:
        """
        执行AND-OR图搜索以找到解决方案。

        返回:
        - Tuple[bool, List[Any]]: 是否找到解决方案以及解决方案路径。
        """
        def or_search(belief: BeliefState, path: List[Any]) -> Tuple[bool, List[Any]]:
            if all(self.goal_test(state) for state in belief.states):
                return True, path
            if belief in path:
                return False, []

            for action in self.actions_fn(belief):
                plan = [(action, and_search(self.predict(belief, action), path + [belief]))]
                if all(sub_plan[0] for _, sub_plan in plan):
                    return True, [action] + [sub_plan[1] for _, sub_plan in plan]
            return False, []

        def and_search(beliefs: List[BeliefState], path: List[Any]) -> Tuple[bool, List[Any]]:
            results = []
            for belief in beliefs:
                success, plan = or_search(belief, path)
                if not success:
                    return False, []
                results.append(plan)
            return True, results

        return or_search(self.initial_belief, [])


class VacuumWorld(PartialObservationsSearch):
    """
    部分观测环境下的非确定性吸尘器世界。
    """
    def __init__(self, initial_state: Tuple[str, Tuple[bool, bool]], goal_test: Callable[[Any], bool]):
        """
        初始化部分观测环境下的非确定性吸尘器世界。

        参数:
        - initial_state (Tuple[str, Tuple[bool, bool]]): 初始状态。
        - goal_test (Callable[[Any], bool]): 测试目标状态的函数。
        """
        initial_belief = BeliefState([initial_state])
        percept_fn = self.percept

        def actions_fn(state: Tuple[str, Tuple[bool, bool]]) -> List[Callable[[Any], Any]]:
            return [self.suck, self.move_left, self.move_right]

        super().__init__(initial_belief, goal_test, actions_fn, percept_fn)

    def percept(self, state: Tuple[str, Tuple[bool, bool]]) -> Tuple[str, bool]:
        """
        返回当前状态的感知信息。

        参数:
        - state (Tuple[str, Tuple[bool, bool]]): 当前状态。

        返回:
        - Tuple[str, bool]: 感知信息。
        """
        return (state[0], state[1][0] if state[0] == 'A' else state[1][1])

    def results(self, state: Tuple[str, Tuple[bool, bool]], action: Callable[[Any], Any]) -> List[Tuple[str, Tuple[bool, bool]]]:
        """
        获取给定状态和操作的所有可能结果。

        参数:
        - state (Tuple[str, Tuple[bool, bool]]): 当前状态。
        - action (Callable[[Any], Any]): 施加于状态的操作。

        返回:
        - List[Tuple[str, Tuple[bool, bool]]]: 所有可能的结果状态。
        """
        np.random.seed(42)  # 为了结果可重复
        if action == self.suck:
            if state[0] == 'A':
                return [('A', (False, state[1][1])), ('A', (False, not state[1][1]))]
            else:
                return [('B', (state[1][0], False)), ('B', (not state[1][0], False))]
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

    # 创建部分观测环境下的吸尘器世界
    vacuum_world = VacuumWorld(initial_state, goal_test)

    # 执行AND-OR图搜索
    solution_found, solution_path = vacuum_world.and_or_search()
    if solution_found:
        print("找到解决方案:", solution_path)
    else:
        print("未找到解决方案。")
