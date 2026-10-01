# 00_2.1.1_Problem-Solving_Agents

"""

Lecture: 2_Problem-solving/2.1_Solving_Problems_by_Searching
Content: 00_2.1.1_Problem-Solving_Agents

"""

import numpy as np
from typing import Tuple

class EigenvalueMethods:
    def __init__(self, matrix: np.ndarray):
        """
        初始化特征值方法类。

        Args:
        - matrix (np.ndarray): 需要进行特征值分解的矩阵。
        """
        assert matrix.shape[0] == matrix.shape[1], "矩阵必须是方阵。"
        self.matrix = matrix
        self.n = matrix.shape[0]

    def power_iteration(self, max_iterations: int = 1000, tol: float = 1e-10) -> Tuple[float, np.ndarray]:
        """
        使用幂迭代法计算矩阵的主特征值及其对应的特征向量。

        Args:
        - max_iterations (int): 最大迭代次数。
        - tol (float): 收敛容差。

        Returns:
        - eigenvalue (float): 主特征值。
        - eigenvector (np.ndarray): 对应的特征向量。
        """
        b_k = np.random.rand(self.n)
        b_k /= np.linalg.norm(b_k)

        for _ in range(max_iterations):
            b_k1 = np.dot(self.matrix, b_k)
            b_k1_norm = np.linalg.norm(b_k1)
            b_k1 /= b_k1_norm

            if np.linalg.norm(b_k1 - b_k) < tol:
                break

            b_k = b_k1

        eigenvalue = np.dot(b_k.T, np.dot(self.matrix, b_k))
        return eigenvalue, b_k

    def inverse_power_iteration(self, shift: float, max_iterations: int = 1000, tol: float = 1e-10) -> Tuple[float, np.ndarray]:
        """
        使用反幂迭代法计算矩阵的最小特征值及其对应的特征向量。

        Args:
        - shift (float): 移位参数。
        - max_iterations (int): 最大迭代次数。
        - tol (float): 收敛容差。

        Returns:
        - eigenvalue (float): 最小特征值。
        - eigenvector (np.ndarray): 对应的特征向量。
        """
        shifted_matrix = self.matrix - shift * np.eye(self.n)
        b_k = np.random.rand(self.n)
        b_k /= np.linalg.norm(b_k)

        for _ in range(max_iterations):
            b_k1 = np.linalg.solve(shifted_matrix, b_k)
            b_k1_norm = np.linalg.norm(b_k1)
            b_k1 /= b_k1_norm

            if np.linalg.norm(b_k1 - b_k) < tol:
                break

            b_k = b_k1

        eigenvalue = np.dot(b_k.T, np.dot(self.matrix, b_k))
        return eigenvalue, b_k

    def qr_algorithm(self, max_iterations: int = 1000, tol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用 QR 算法计算矩阵的所有特征值和特征向量。

        Args:
        - max_iterations (int): 最大迭代次数。
        - tol (float): 收敛容差。

        Returns:
        - eigenvalues (np.ndarray): 矩阵的特征值。
        - eigenvectors (np.ndarray): 对应的特征向量。
        """
        A = self.matrix.copy()
        Q_total = np.eye(self.n)

        for _ in range(max_iterations):
            Q, R = np.linalg.qr(A)
            A = R @ Q
            Q_total = Q_total @ Q

            if np.allclose(A - np.diag(np.diag(A)), 0, atol=tol):
                break

        eigenvalues = np.diag(A)
        return eigenvalues, Q_total

    def jacobi_method(self, max_iterations: int = 1000, tol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用 Jacobi 方法计算矩阵的所有特征值和特征向量。

        Args:
        - max_iterations (int): 最大迭代次数。
        - tol (float): 收敛容差。

        Returns:
        - eigenvalues (np.ndarray): 矩阵的特征值。
        - eigenvectors (np.ndarray): 对应的特征向量。
        """
        A = self.matrix.copy()
        V = np.eye(self.n)

        for iteration in range(max_iterations):
            max_val = 0
            p = q = 0

            for i in range(self.n):
                for j in range(i + 1, self.n):
                    if abs(A[i, j]) > max_val:
                        max_val = abs(A[i, j])
                        p, q = i, j

            if max_val < tol:
                break

            if A[p, p] != A[q, q]:
                phi = 0.5 * np.arctan(2 * A[p, q] / (A[q, q] - A[p, p]))
            else:
                phi = np.pi / 4

            c = np.cos(phi)
            s = np.sin(phi)

            J = np.eye(self.n)
            J[p, p] = c
            J[q, q] = c
            J[p, q] = s
            J[q, p] = -s

            A = J.T @ A @ J
            V = V @ J

        eigenvalues = np.diag(A)
        return eigenvalues, V

# 示例矩阵
matrix = np.array([
    [4, 1, 2],
    [1, 2, 0],
    [2, 0, 3]
])

solver = EigenvalueMethods(matrix)

# 使用幂迭代法计算主特征值及其对应的特征向量
eigenvalue, eigenvector = solver.power_iteration()
print("主特征值:", eigenvalue)
print("对应的特征向量:\n", eigenvector)

# 使用反幂迭代法计算最小特征值及其对应的特征向量
eigenvalue, eigenvector = solver.inverse_power_iteration(shift=1.0)
print("最小特征值:", eigenvalue)
print("对应的特征向量:\n", eigenvector)

# 使用 QR 算法计算所有特征值和特征向量
eigenvalues, eigenvectors = solver.qr_algorithm()
print("特征值 (QR 算法):", eigenvalues)
print("特征向量 (QR 算法):\n", eigenvectors)

# 使用 Jacobi 方法计算所有特征值和特征向量
eigenvalues, eigenvectors = solver.jacobi_method()
print("特征值 (Jacobi 方法):", eigenvalues)
print("特征向量 (Jacobi 方法):\n", eigenvectors)
