import torch 
import matplotlib.pyplot as plt
from problem import exact_solution

# torch.linspace(start, end, steps)：
# 在 [-1, 1] 区间内生成 201 个等间距的 x 坐标，形状为 (201,)
x = torch.linspace(-1, 1, 201)
# torch.reshape(input, shape)：
# 将 x 重塑为 (201, 1)，表示 201 个样本，每个样本包含 1 个 x 特征
x = torch.reshape(x, (201, 1),)  # shape: (201, 1)
alpha = 1.0
t_values = torch.tensor([0.0, 0.1, 0.5])
# , dtype=torch.float32).view(-1, 1)  # 将 3 个时间值转换为形状为 (3, 1) 的 float32 张量


for t in t_values:
    # 将当前时间值复制成与 x 相同的形状
    t_batch = torch.full_like(x, t.item())

    # 计算当前时间下所有 x 位置的精确解
    u = exact_solution(t_batch, x, alpha)

    # 绘制曲线，并将当前时间显示在图例中
    plt.plot(
        x.squeeze(),
        u.squeeze(),
        label=f't={t.item():.1f}'
    )

# 设置横坐标名称为 x，表示空间位置
plt.xlabel('x')
# 设置纵坐标名称为 u，表示精确解的数值
plt.ylabel('u')
# 设置图像标题
plt.title('Exact Solution')
# 显示图例，用来区分不同时间 t 对应的曲线
plt.legend()
# 显示绘制好的图像窗口
plt.show()
