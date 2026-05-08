import numpy as np
from scipy.optimize import minimize
# ---------------------
# 1. Rastrigin 函数（全局最小值 f(0,...,0) = 0）
# ---------------------
def rastrigin(x):
    A = 10.0
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2.0 * np.pi * x))
# ---------------------
# 2. 延拓法（continuation method）求解
#    核心思路：从平滑二次函数（alpha=0）逐步过渡到完整 Rastrigin（alpha=1）
#    每一步用 L-BFGS-B 跟踪全局最小值，避免陷入局部最优
# ---------------------
def solve_rastrigin(dim, bounds=(-5.12, 5.12), n_steps=30):
    # 题目指定初始点 [5, 5, ..., 5]
    x = np.full(dim, 5.0)
    bnds = [bounds] * dim
    # 逐步增大余弦项幅度：alpha 从 0 到 1
    for alpha in np.linspace(0.0, 1.0, n_steps):
        def obj(x, a=alpha):
            A = 10.0
            return A * len(x) + np.sum(x**2 - a * A * np.cos(2.0 * np.pi * x))

        res = minimize(obj, x, method='L-BFGS-B', bounds=bnds,
                       options={'maxiter': 20000, 'ftol': 1e-16, 'gtol': 1e-16})
        x = res.x
    # 在完整 Rastrigin 上做最终高精度收敛
    res = minimize(rastrigin, x, method='L-BFGS-B', bounds=bnds,
                   options={'maxiter': 50000, 'ftol': 1e-16, 'gtol': 1e-16})
    return res.x, res.fun
# ---------------------
# 3. 运行 2 / 10 / 30 维
# ---------------------
if __name__ == "__main__":
    for dim in [2, 10, 30]:
        x_opt, f_opt = solve_rastrigin(dim)
        print(f"===== {dim:2d}-D Rastrigin =====")
        print(f"f(x*) = {f_opt:.14e}")
        if dim <= 10:
            print(f"x*    = {np.round(x_opt, 12)}")
        else:
            print(f"x* (first 10) = {np.round(x_opt[:10], 12)}")
        print()
