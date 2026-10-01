# 01_2.4.2_Constraint_Propagation:_Inference_in_CSPs

"""

Lecture: 2_Problem-solving/2.4_Constraint_Satisfaction_Problems
Content: 01_2.4.2_Constraint_Propagation:_Inference_in_CSPs

"""

import numpy as np
from typing import List, Tuple, Dict, Any, Set

class CSP:
    def __init__(self, variables: List[Any], domains: Dict[Any, List[Any]], constraints: Dict[Tuple[Any, Any], Set[Tuple[Any, Any]]]):
        """
        初始化CSP实例。

        参数:
        - variables: 变量列表。
        - domains: 每个变量的取值域。
        - constraints: 变量间的约束条件。
        """
        self.variables = variables
        self.domains = domains
        self.constraints = constraints

    def is_consistent(self, var: Any, value: Any, assignment: Dict[Any, Any]) -> bool:
        """
        检查赋值是否一致。

        参数:
        - var: 变量。
        - value: 变量的值。
        - assignment: 当前的赋值。

        返回:
        - 如果一致返回True，否则返回False。
        """
        for (xi, xj) in self.constraints:
            if xi == var and xj in assignment:
                if (value, assignment[xj]) not in self.constraints[xi, xj]:
                    return False
            if xj == var and xi in assignment:
                if (assignment[xi], value) not in self.constraints[xi, xj]:
                    return False
        return True

    def revise(self, xi: Any, xj: Any) -> bool:
        """
        修订变量xi的域，使其与变量xj的一致。

        参数:
        - xi: 变量xi。
        - xj: 变量xj。

        返回:
        - 如果xi的域被修订返回True，否则返回False。
        """
        revised = False
        for x in self.domains[xi]:
            if all((x, y) not in self.constraints[xi, xj] for y in self.domains[xj]):
                self.domains[xi].remove(x)
                revised = True
        return revised

    def ac3(self) -> bool:
        """
        执行AC-3算法，使CSP达到弧一致性。

        返回:
        - 如果成功返回True，否则返回False。
        """
        queue = [(xi, xj) for xi in self.variables for xj in self.variables if xi != xj and (xi, xj) in self.constraints]
        while queue:
            (xi, xj) = queue.pop(0)
            if self.revise(xi, xj):
                if not self.domains[xi]:
                    return False
                for xk in self.variables:
                    if xk != xi and xk != xj and (xk, xi) in self.constraints:
                        queue.append((xk, xi))
        return True

# 示例用法
if __name__ == "__main__":
    # 定义变量和域
    variables = ['A', 'B', 'C']
    domains = {
        'A': [1, 2, 3],
        'B': [1, 2, 3],
        'C': [1, 2, 3]
    }

    # 定义约束
    constraints = {
        ('A', 'B'): {(1, 2), (2, 3), (3, 1)},
        ('B', 'A'): {(2, 1), (3, 2), (1, 3)},
        ('A', 'C'): {(1, 3), (2, 1), (3, 2)},
        ('C', 'A'): {(3, 1), (1, 2), (2, 3)},
        ('B', 'C'): {(1, 2), (2, 3), (3, 1)},
        ('C', 'B'): {(2, 1), (3, 2), (1, 3)}
    }

    # 创建CSP实例
    csp = CSP(variables, domains, constraints)

    # 执行AC-3算法
    if csp.ac3():
        print("弧一致性已达到")
        print("修订后的域:", csp.domains)
    else:
        print("无法达到弧一致性")
