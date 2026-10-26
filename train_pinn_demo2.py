import torch
from problem import exact_solution
import matplotlib.pyplot as plt


def sample_interior(n):

    # 这两段是关键，调用torch.rand()函数生成随机数，生成的随机数在[0,1)区间内均匀分布。
    t = torch.rand(n, 1)  # 在 [0, 1] 区间内随机采样时间
    x = torch.rand(n, 1) * 2 - 1  # 在 [-1, 1] 区间内随机采样空间位置
    return t, x


def sample_initial(n):

    x = torch.rand(n, 1) * 2 - 1  # 在 [-1, 1] 区间内随机采样空间位置
    t = torch.zeros_like(x)  # 初始时间 t=0
    return t, x


def sample_boundary(n):

    t = torch.rand(n, 1)  # 在 [0, 1] 区间内随机采样时间
    x_left = torch.full_like(t, -1.0)  # 左边界 x=-1
    x_right = torch.full_like(t, 1.0)  # 右边界
    return t, x_left, x_right


def L_PDE(residual):
    return torch.mean(residual**2)


def L_IC(u_pred_ic, x_ic):
    u_exact_ic = torch.sin(torch.pi * x_ic)
    return torch.mean((u_pred_ic - u_exact_ic) ** 2)


def L_BC(u_pred_bl, u_pred_br):
    return torch.mean(u_pred_bl**2) + torch.mean(u_pred_br**2)


def L_total(loss_pde, loss_ic, loss_bc):
    return loss_pde + 0.5 * loss_ic + 0.5 * loss_bc


n_samples = 100

if __name__ == "__main__":

    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)
    net = torch.nn.Sequential(
        torch.nn.Linear(2, 20),
        torch.nn.Tanh(),
        torch.nn.Linear(20, 1),
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    for step in range(2000):
        optimizer.zero_grad()
        t, x = sample_interior(n=n_samples)
        t_ic, x_ic = sample_initial(n=n_samples)
        t_bc, x_left, x_right = sample_boundary(n=n_samples)

        t.requires_grad_(True)
        x.requires_grad_(True)

        tx = torch.cat((t, x), dim=1)
        tx_ic = torch.cat((t_ic, x_ic), dim=1)
        txl_bc = torch.cat((t_bc, x_left), dim=1)
        txr_bc = torch.cat((t_bc, x_right), dim=1)

        u_pred = net(tx)
        u_pred_ic = net(tx_ic)
        u_pred_bl = net(txl_bc)
        u_pred_br = net(txr_bc)

        u_t = torch.autograd.grad(
            outputs=u_pred,
            inputs=t,
            grad_outputs=torch.ones_like(u_pred),
            create_graph=True,
        )[0]

        u_x = torch.autograd.grad(
            outputs=u_pred,
            inputs=x,
            grad_outputs=torch.ones_like(u_pred),
            create_graph=True,
        )[0]

        u_xx = torch.autograd.grad(
            outputs=u_x,
            inputs=x,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
        )[0]

        alpha = 1.0
        residual = u_t - alpha * u_xx

        loss_pde = L_PDE(residual)
        loss_ic = L_IC(u_pred_ic, x_ic)
        loss_bc = L_BC(u_pred_bl, u_pred_br)
        loss_total = L_total(loss_pde, loss_ic, loss_bc)

        loss_total.backward()
        optimizer.step()

        if (step + 1) % 200 == 0:
            print(
                f"Step {step}, Loss PDE: {loss_pde.item():.6f}, Loss IC: {loss_ic.item():.6f}, Loss BC: {loss_bc.item():.6f}, Total Loss: {loss_total.item():.6f}"
            )

    net.eval()
    with torch.no_grad():
        # 在 [-1, 1] 内生成 201 个空间位置
        x = torch.linspace(-1, 1, 201).view(-1, 1)

        t = torch.zeros_like(x)

        # 将每个样本组合成 [t, x]
        tx = torch.cat([t, x], dim=1)

        # 使用训练完成后的网络进行预测
        u_pred = net(tx)

        # 设置热扩散系数
        alpha = 1.0

        # 计算解析解
        u_exact = exact_solution(t, x, alpha=alpha)

        # 创建图像
        plt.figure(figsize=(10, 6))

        # 创建第一个子图
        plt.subplot(1, 2, 1)

        # 绘制预测解
        plt.plot(
            x.numpy(),
            u_pred.numpy(),
            label="Predicted",
        )

        # 绘制解析解
        plt.plot(
            x.numpy(),
            u_exact.numpy(),
            label="Exact",
        )

        # 设置坐标轴和标题
        plt.xlabel("x")
        plt.ylabel("u")
        plt.title("Solution at t=0.0")

        # 显示图例
        plt.legend()

        # 计算并打印最大绝对值
        max_abs = torch.max(torch.abs(u_pred - u_exact))
        print("Maximum absolute error at t=0.0", max_abs.item())

        # 创建网格
        t_axis = torch.linspace(0, 1, 101)
        x_axis = torch.linspace(-1, 1, 201)
        t_grid, x_grid = torch.meshgrid(t_axis, x_axis, indexing="ij")
        t_eval = t_grid.reshape(-1, 1)
        x_eval = x_grid.reshape(-1, 1)
        tx_eval = torch.cat([t_eval, x_eval], dim=1)
        u_pred_eval = net(tx_eval).reshape(t_grid.shape)
        u_exact_eval = exact_solution(t_eval, x_eval, alpha=alpha).reshape(t_grid.shape)
        error_eval = torch.abs(u_pred_eval - u_exact_eval)
        MAE = torch.mean(error_eval)
        relative_L2 = torch.sqrt(torch.sum(error_eval**2) / torch.sum(u_exact_eval**2))
        print("Mean absolute error over the grid:", MAE.item())
        print("Relative L2 error over the grid:", relative_L2.item())

        u_left = u_pred_eval[:, 0]
        u_right = u_pred_eval[:, -1]
        max_abs_left = torch.max(torch.abs(u_left))
        max_abs_right = torch.max(torch.abs(u_right))
    
        # assert max_abs.item() < 0.05, "max_abs is too large"
        print("Maximum absolute value at boundaries:", max_abs_left.item(), max_abs_right.item())

        # 显示图像
        plt.show()
