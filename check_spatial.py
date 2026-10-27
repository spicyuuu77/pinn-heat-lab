# import torch

# x = torch.linspace(-1, 1, 5)
# h = x[2] - x[1]

# def u(x):
#     return x**2

# u_values = u(x)
# u_xx=(u_values[2:]-2*u_values[1:-1]+u_values[:-2])/h**2

# print(u_xx)

# N = len(x)
# D2 = torch.zeros(N - 2, N)

# for i in range(N - 2):
#     D2[i, i] = 1.0
#     D2[i, i + 1] = -2.0
#     D2[i, i + 2] = 1.0

# D2 = D2 / h**2
# print(D2)

# u_xx_matrix = D2 @ u_values
# print(u_xx_matrix)

import torch

for N in (50, 100, 200):
    x = torch.linspace(-1, 1, N, dtype=torch.float64)
    h = x[1] - x[0]

    D2 = torch.zeros(N - 2, N, dtype=x.dtype)

    for i in range(N - 2):
        # 把你已经写对的三行矩阵系数赋值放这里
        D2[i, i] = 1.0
        D2[i, i + 1] = -2.0
        D2[i, i + 2] = 1.0

    # 填完整个矩阵后，统一除以 h 的平方
    D2 = D2 / h**2
    print("N =", N, "矩阵形状 =", D2.shape)

    # 检查三种多项式
    for name, values, expected in [
        ("constant", torch.ones_like(x), 0.0),
        ("linear", x, 0.0),
        ("quadratic", x**2, 2.0),
    ]:
        error = (D2 @ values - expected).abs().max().item()
        print(name, "最大误差 =", error)

    # 检查正弦函数：下面三行由你完成
    u_values = torch.sin(torch.pi * x)
    u_xx_matrix = D2 @ u_values
    u_xx_exact = -(torch.pi ** 2) * torch.sin(torch.pi * x[1:-1])

    sine_error = (u_xx_matrix - u_xx_exact).abs().max().item()
    print("sine 最大误差 =", sine_error)
    print()
