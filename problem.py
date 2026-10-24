import torch


# 定义精确解函数。
# t：时间；x：空间位置；alpha：热扩散系数，默认值为 1.0
def exact_solution(t, x, alpha=1.0):
    # 计算 PDE 的精确解 u(t, x) = exp(-alpha * pi² * t) * sin(pi * x)
    # 计算 exp(-alpha * pi² * t) * sin(pi * x)
    # torch.exp(...) 计算自然指数 e 的幂
    # torch.sin(...) 计算正弦值
    return torch.exp(-alpha * (torch.pi ** 2) * t) * torch.sin(torch.pi * x)


# 只有直接运行本文件时，下面的代码才会执行。
# 如果本文件被其他文件 import，这部分代码不会自动执行。
if __name__ == "__main__":
    # 创建时间张量 t = 0.1。
    # [[0.1]] 先创建一个包含单个元素的二维列表。
    # dtype=torch.float32 指定使用 32 位浮点数。
    # view(-1, 1) 将形状整理为 (1, 1)：
    # 1 行数据，每行包含 1 个特征。
    input_t = torch.tensor([[0.1]], dtype=torch.float32).view(-1, 1)

    # 创建空间位置张量 x = 0.5，写法和 input_t 相同。
    input_x = torch.tensor([[0.5]], dtype=torch.float32).view(-1, 1)

    # 设置热扩散系数 alpha = 2.0。
    alpha = 2.0

    # 调用 exact_solution(input_t, input_x, alpha) 计算精确解。
    # item() 将只包含一个元素的 Tensor 转换为普通 Python 数值，便于输出。
    # f"..." 是格式化字符串，{} 中可以直接放变量或表达式。
    print(f"Exact solution at t={input_t.item()}, x={input_x.item()}: {exact_solution(input_t, input_x, alpha).item()}")



