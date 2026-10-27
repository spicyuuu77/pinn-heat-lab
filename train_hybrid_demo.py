import torch
from problem import exact_solution
import matplotlib.pyplot as plt
from pathlib import Path
from time import perf_counter

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


history_pde = []
history_ic = []
history_bc = []
history_total = []


n_samples = 100  # 初始条件和边界条件的采样数量
n_steps = 5000  # 本轮先做短训练
N = 50  # 每组空间网格点数
n_times = 4  # 每一步抽取四个时间
alpha = 1.0

if __name__ == "__main__":

    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)
    net = torch.nn.Sequential(
        torch.nn.Linear(2, 50),
        torch.nn.Tanh(),
        torch.nn.Linear(50, 50),
        torch.nn.Tanh(),
        torch.nn.Linear(50, 50),
        torch.nn.Tanh(),
        torch.nn.Linear(50, 50),
        torch.nn.Tanh(),
        torch.nn.Linear(50, 50),
        torch.nn.Tanh(),
        torch.nn.Linear(50, 50),
        torch.nn.Tanh(),
        torch.nn.Linear(50, 1),
    )

    print("Output shape:", net(torch.zeros(4, 2)).shape)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

    x_axis = torch.linspace(-1, 1, N)
    h = x_axis[1] - x_axis[0]

    D2 = torch.zeros(N - 2, N, dtype=x_axis.dtype)
    for i in range(N - 2):
        # 把你已经写对的三行矩阵系数赋值放这里
        D2[i, i] = 1.0
        D2[i, i + 1] = -2.0
        D2[i, i + 2] = 1.0

        # 填完整个矩阵后，统一除以 h 的平方
    D2 = D2 / h**2
    assert torch.isfinite(D2).all().item(), "D2 contains NaN or Inf"

    start_time = perf_counter()
    for step in range(n_steps):
        optimizer.zero_grad()

        times = torch.rand(n_times)
        t = times.repeat_interleave(N).reshape(-1, 1)
        x = x_axis.repeat(n_times).reshape(-1, 1)
        t_ic, x_ic = sample_initial(n=n_samples)
        t_bc, x_left, x_right = sample_boundary(n=n_samples)

        t.requires_grad_(True)

        # 转为矩阵，输入net
        tx = torch.cat((t, x), dim=1)
        tx_ic = torch.cat((t_ic, x_ic), dim=1)
        txl_bc = torch.cat((t_bc, x_left), dim=1)
        txr_bc = torch.cat((t_bc, x_right), dim=1)

        # net生成预测值
        u_pred = net(tx)
        u_pred_ic = net(tx_ic)
        u_pred_bl = net(txl_bc)
        u_pred_br = net(txr_bc)

        # 迭代求导数
        u_t = torch.autograd.grad(
            outputs=u_pred,
            inputs=t,
            grad_outputs=torch.ones_like(u_pred),
            create_graph=True,
        )[0]
        U = u_pred.reshape(n_times, N)
        Ut = u_t.reshape(n_times, N)
        Uxx = U @ D2.T

        residual = Ut[:, 1:-1] - alpha * Uxx

        # 计算损失函数
        loss_pde = L_PDE(residual)
        loss_ic = L_IC(u_pred_ic, x_ic)
        loss_bc = L_BC(u_pred_bl, u_pred_br)
        loss_total = L_total(loss_pde, loss_ic, loss_bc)

        # 记录每个损失函数的历史值
        history_pde.append(loss_pde.item())
        history_ic.append(loss_ic.item())
        history_bc.append(loss_bc.item())
        history_total.append(loss_total.item())

        # 反向传播和优化
        if step == 0:
            print("U、Ut：", U.shape, Ut.shape)
            print("Uxx、residual：", Uxx.shape, residual.shape)

            pde_grad = torch.autograd.grad(
                loss_pde,
                net[0].weight,
                retain_graph=True,
            )[0]
            print("PDE 对第一层权重的梯度范数：", pde_grad.norm().item())
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        if (step + 1) % 200 == 0:
            print("Next learning rate:", scheduler.get_last_lr()[0])
            print(
                f"Step {step}, Loss PDE: {loss_pde.item():.6f}, Loss IC: {loss_ic.item():.6f}, Loss BC: {loss_bc.item():.6f}, Total Loss: {loss_total.item():.6f}"
            )

    elapsed = perf_counter() - start_time
    print(f"训练耗时：{elapsed:.2f} 秒")

    history = {
        "pde": history_pde,
        "ic": history_ic,
        "bc": history_bc,
        "total": history_total,
    }
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    torch.save(history, output_dir / f"hybrid_loss_seed42_steps{n_steps}.pt")
    model_path = output_dir / f"hybrid_model_seed42_steps{n_steps}.pt"
    torch.save(net.state_dict(), model_path)
    print("Model saved to:", model_path)
    net.eval()
    with torch.no_grad():
        # 在 [-1, 1] 内生成 201 个空间位置
        x = torch.linspace(-1, 1, 201).view(-1, 1)

        t = torch.zeros_like(x)

        # 将每个样本组合成 [t, x]
        tx = torch.cat([t, x], dim=1)

        # 使用训练完成后的网络进行预测
        u_pred = net(tx)

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

        # 分别在时间方向和空间方向上建立规则采样点。
        # t_axis 的形状是 [101]，表示 0 到 1 之间的 101 个时间点；
        # x_axis 的形状是 [201]，表示 -1 到 1 之间的 201 个空间点。
        # 评估网格比训练时的随机采样更规则，便于全面检查网络在整个区域内的表现。
        t_axis = torch.linspace(0, 1, 101)
        x_axis = torch.linspace(-1, 1, 201)

        # 将所有时间点和所有空间点两两组合，形成二维时空网格。
        # indexing="ij" 保证第 0 维对应时间 t，第 1 维对应空间 x。
        # 因此 t_grid 和 x_grid 的形状都是 [101, 201]：
        #   t_grid[i, j] = 第 i 个时间点
        #   x_grid[i, j] = 第 j 个空间点
        # 网格一共包含 101 * 201 = 20301 个时空位置。
        t_grid, x_grid = torch.meshgrid(t_axis, x_axis, indexing="ij")

        # 神经网络的输入需要是二维张量，每一行代表一个样本 [t, x]。
        # reshape(-1, 1) 把 [101, 201] 网格展平成 [20301, 1] 的列向量；
        # -1 表示让 PyTorch 自动计算这一维的长度。
        t_eval = t_grid.reshape(-1, 1)
        x_eval = x_grid.reshape(-1, 1)

        # 将时间列和空间列按列拼接，得到网络所需的输入：
        # tx_eval.shape = [20301, 2]，每一行都是一个 [t, x] 坐标。
        tx_eval = torch.cat([t_eval, x_eval], dim=1)

        # 对全部 20301 个时空点进行预测。
        # 网络输出原本是 [20301, 1]，再还原为 [101, 201]，
        # 这样第一个下标对应时间，第二个下标对应空间，方便后续分析或绘图。
        u_pred_eval = net(tx_eval).reshape(t_grid.shape)

        # 在完全相同的时空位置上计算解析解，作为误差计算的基准。
        # exact_solution 接收展平后的 [20301, 1] 坐标，因此结果也先是 [20301, 1]；
        # reshape 后与 u_pred_eval 保持相同的 [101, 201] 形状。
        u_exact_eval = exact_solution(t_eval, x_eval, alpha=alpha).reshape(t_grid.shape)

        # 计算每个网格点的绝对误差：
        # error_eval[i, j] = |网络预测值 - 解析解|。
        # 误差越接近 0，说明该时空位置的预测越准确。
        error_eval = torch.abs(u_pred_eval - u_exact_eval)

        # MAE（Mean Absolute Error，平均绝对误差）：
        # 对全部时空网格点的绝对误差取平均，反映整体平均误差大小。
        MAE = torch.mean(error_eval)

        # Relative L2 error（相对 L2 误差）：
        # 分子是预测误差的 L2 范数，分母是解析解的 L2 范数。
        # 它衡量误差相对于真实解整体规模的大小，通常比单独看绝对误差更有可比性。
        relative_L2 = torch.sqrt(torch.sum(error_eval**2) / torch.sum(u_exact_eval**2))
        print("Mean absolute error over the grid:", MAE.item())
        print("Relative L2 error over the grid:", relative_L2.item())

        # 取出预测结果在两个空间边界上的所有时间值：
        #   u_pred_eval[:, 0]  对应左边界 x=-1；
        #   u_pred_eval[:, -1] 对应右边界 x=1。
        # 冒号表示保留全部 101 个时间点，0 和 -1 分别表示第一列和最后一列。
        u_left = u_pred_eval[:, 0]
        u_right = u_pred_eval[:, -1]

        # 计算边界预测值的最大绝对值。
        # 本问题的边界条件是 u(t, -1)=0 和 u(t, 1)=0，
        # 所以这两个数越接近 0，说明网络越满足边界条件。
        max_abs_left = torch.max(torch.abs(u_left))
        max_abs_right = torch.max(torch.abs(u_right))

        # assert max_abs.item() < 0.05, "max_abs is too large"
        print(
            "Maximum absolute value at boundaries:",
            max_abs_left.item(),
            max_abs_right.item(),
        )

        plt.subplot(1, 2, 2)
        plt.plot(range(1, n_steps + 1), history_pde, label="PDE")
        plt.plot(range(1, n_steps + 1), history_ic, label="IC")
        plt.plot(range(1, n_steps + 1), history_bc, label="BC")
        plt.plot(range(1, n_steps + 1), history_total, label="Total")
        plt.yscale("log")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()

        # 显示图像
        plt.show()
