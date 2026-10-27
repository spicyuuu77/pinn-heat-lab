import torch
from problem import exact_solution

alpha = 1.0
N = 50

times = torch.tensor([0.0, 0.1, 0.5], dtype=torch.float64)
n_times = len(times)

x_axis = torch.linspace(-1, 1, N, dtype=torch.float64)
h = x_axis[1] - x_axis[0]

x = x_axis.repeat(n_times).reshape(-1, 1)
t = times.repeat_interleave(N).reshape(-1, 1)
t.requires_grad_(True)

D2 = torch.zeros(N - 2, N, dtype=x_axis.dtype)

for i in range(N - 2):
    # 把你已经写对的三行矩阵系数赋值放这里
    D2[i, i] = 1.0
    D2[i, i + 1] = -2.0
    D2[i, i + 2] = 1.0

    # 填完整个矩阵后，统一除以 h 的平方
D2 = D2 / h**2
print("N =", N, "矩阵形状 =", D2.shape)

u = exact_solution(t, x, alpha)
u_t = torch.autograd.grad(
    outputs=u,
    inputs=t,
    grad_outputs=torch.ones_like(u),
    create_graph=True,
)[0]

U = u.reshape(n_times, N)
Ut = u_t.reshape(n_times, N)
Uxx = U @ D2.T

residual = Ut[:, 1:-1] - alpha * Uxx

print("t、x：", t.shape, x.shape)
print("U、Ut：", U.shape, Ut.shape)
print("D2：", D2.shape)
print("Uxx、residual：", Uxx.shape, residual.shape)

print("每个时间的最大残差：")
print(residual.abs().max(dim=1).values)
