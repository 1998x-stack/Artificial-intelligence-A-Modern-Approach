# 01_2.2.2_Local_Search_in_Continuous_Spaces

"""

Lecture: 2_Problem-solving/2.2_Beyond_Classical_Search
Content: 01_2.2.2_Local_Search_in_Continuous_Spaces

"""

import numpy as np
from typing import Callable, Tuple, Any

class LocalSearchContinuous:
    """
    连续空间本地搜索算法的基类。
    """
    def __init__(self, initial_state: np.ndarray, objective_function: Callable[[np.ndarray], float]):
        """
        初始化本地搜索算法。

        参数:
        - initial_state (np.ndarray): 算法的初始状态。
        - objective_function (Callable[[np.ndarray], float]): 目标函数。
        """
        self.current_state = initial_state
        self.objective_function = objective_function

    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        优化目标函数的抽象方法。

        返回:
        - Tuple[np.ndarray, float]: 优化后的状态及其对应的目标函数值。
        """
        raise NotImplementedError("子类应该实现 optimize 方法。")


class GradientAscent(LocalSearchContinuous):
    """
    梯度上升算法。
    """
    def optimize(self, learning_rate: float = 0.01, max_iterations: int = 1000) -> Tuple[np.ndarray, float]:
        """
        使用梯度上升算法优化目标函数。

        参数:
        - learning_rate (float): 学习率，控制每步更新幅度。
        - max_iterations (int): 最大迭代次数。

        返回:
        - Tuple[np.ndarray, float]: 优化后的状态及其对应的目标函数值。
        """
        for _ in range(max_iterations):
            gradient = self._compute_gradient(self.current_state)
            self.current_state += learning_rate * gradient
        best_value = self.objective_function(self.current_state)
        return self.current_state, best_value

    def _compute_gradient(self, state: np.ndarray) -> np.ndarray:
        """
        计算给定状态的梯度。

        参数:
        - state (np.ndarray): 当前状态。

        返回:
        - np.ndarray: 计算出的梯度。
        """
        epsilon = 1e-8
        gradient = np.zeros_like(state)
        for i in range(len(state)):
            state_epsilon = np.array(state, copy=True)
            state_epsilon[i] += epsilon
            gradient[i] = (self.objective_function(state_epsilon) - self.objective_function(state)) / epsilon
        return gradient


class SimulatedAnnealing(LocalSearchContinuous):
    """
    模拟退火算法。
    """
    def optimize(self, initial_temperature: float = 1.0, cooling_rate: float = 0.95, max_iterations: int = 1000) -> Tuple[np.ndarray, float]:
        """
        使用模拟退火算法优化目标函数。

        参数:
        - initial_temperature (float): 初始温度。
        - cooling_rate (float): 降温速率。
        - max_iterations (int): 最大迭代次数。

        返回:
        - Tuple[np.ndarray, float]: 优化后的状态及其对应的目标函数值。
        """
        temperature = initial_temperature
        for _ in range(max_iterations):
            neighbor = self.current_state + np.random.normal(size=self.current_state.shape)
            delta_e = self.objective_function(neighbor) - self.objective_function(self.current_state)
            if delta_e > 0 or np.random.rand() < np.exp(delta_e / temperature):
                self.current_state = neighbor
            temperature *= cooling_rate
        best_value = self.objective_function(self.current_state)
        return self.current_state, best_value


class NewtonRaphson(LocalSearchContinuous):
    """
    牛顿-拉夫森法。
    """
    def optimize(self, max_iterations: int = 1000) -> Tuple[np.ndarray, float]:
        """
        使用牛顿-拉夫森法优化目标函数。

        参数:
        - max_iterations (int): 最大迭代次数。

        返回:
        - Tuple[np.ndarray, float]: 优化后的状态及其对应的目标函数值。
        """
        for _ in range(max_iterations):
            gradient = self._compute_gradient(self.current_state)
            hessian = self._compute_hessian(self.current_state)
            if np.linalg.det(hessian) == 0:
                print("Hessian matrix is singular.")
                break
            self.current_state -= np.linalg.inv(hessian).dot(gradient)
        best_value = self.objective_function(self.current_state)
        return self.current_state, best_value

    def _compute_gradient(self, state: np.ndarray) -> np.ndarray:
        """
        计算给定状态的梯度。

        参数:
        - state (np.ndarray): 当前状态。

        返回:
        - np.ndarray: 计算出的梯度。
        """
        epsilon = 1e-8
        gradient = np.zeros_like(state)
        for i in range(len(state)):
            state_epsilon = np.array(state, copy=True)
            state_epsilon[i] += epsilon
            gradient[i] = (self.objective_function(state_epsilon) - self.objective_function(state)) / epsilon
        return gradient

    def _compute_hessian(self, state: np.ndarray) -> np.ndarray:
        """
        计算给定状态的Hessian矩阵。

        参数:
        - state (np.ndarray): 当前状态。

        返回:
        - np.ndarray: 计算出的Hessian矩阵。
        """
        epsilon = 1e-5
        hessian = np.zeros((len(state), len(state)))
        for i in range(len(state)):
            for j in range(len(state)):
                state_epsilon_i = np.array(state, copy=True)
                state_epsilon_j = np.array(state, copy=True)
                state_epsilon_ij = np.array(state, copy=True)

                state_epsilon_i[i] += epsilon
                state_epsilon_j[j] += epsilon
                state_epsilon_ij[i] += epsilon
                state_epsilon_ij[j] += epsilon

                f_x = self.objective_function(state)
                f_x_i = self.objective_function(state_epsilon_i)
                f_x_j = self.objective_function(state_epsilon_j)
                f_x_ij = self.objective_function(state_epsilon_ij)

                hessian[i, j] = (f_x_ij - f_x_i - f_x_j + f_x) / (epsilon ** 2)
        return hessian

# 示例用法：
if __name__ == "__main__":
    # 定义一个示例目标函数（最小化一个二次函数）
    def objective_function(x: np.ndarray) -> float:
        return np.sum(x**2)

    # 算法的初始状态
    initial_state = np.random.rand(5)

    # 梯度上升优化
    gradient_ascent = GradientAscent(initial_state, objective_function)
    optimized_state, best_value = gradient_ascent.optimize()
    print(f"梯度上升: 优化状态 = {optimized_state}, 最优值 = {best_value}")

    # 模拟退火优化
    simulated_annealing = SimulatedAnnealing(initial_state, objective_function)
    optimized_state, best_value = simulated_annealing.optimize()
    print(f"模拟退火: 优化状态 = {optimized_state}, 最优值 = {best_value}")

    # 牛顿-拉夫森法优化
    newton_raphson = NewtonRaphson(initial_state, objective_function)
    optimized_state, best_value = newton_raphson.optimize()
    print(f"牛顿-拉夫森法: 优化状态 = {optimized_state}, 最优值 = {best_value}")
