import json
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import torch

from problem import exact_solution


# 对照只切换空间求导方式；网络、物理参数、采样和训练预算保持一致。
method = "standard"
seed = 42
n_steps = 5000
N = 50
n_times = 4
n_samples = 100
alpha = 1.0


def sample_initial(n):
    x = torch.rand(n, 1) * 2 - 1
    t = torch.zeros_like(x)
    return t, x


def sample_boundary(n):
    t = torch.rand(n, 1)
    x_left = torch.full_like(t, -1.0)
    x_right = torch.full_like(t, 1.0)
    return t, x_left, x_right


def L_PDE(residual):
    return torch.mean(residual**2)


def L_IC(u_pred_ic, x_ic):
    return torch.mean((u_pred_ic - torch.sin(torch.pi * x_ic)) ** 2)


def L_BC(u_pred_bl, u_pred_br):
    return torch.mean(u_pred_bl**2) + torch.mean(u_pred_br**2)


def L_total(loss_pde, loss_ic, loss_bc):
    return loss_pde + 0.5 * loss_ic + 0.5 * loss_bc


def train(method, seed, n_steps, N, n_times, n_samples, alpha):
    if method not in ("standard", "hybrid"):
        raise ValueError('method must be "standard" or "hybrid"')
    if N < 3 or min(n_steps, n_times, n_samples) < 1:
        raise ValueError("N must be at least 3; training counts must be positive")

    # 每次重新设种子并创建网络；两分支保持相同的随机数调用顺序。
    torch.manual_seed(seed)
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
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1000, gamma=0.5
    )
    history = {"pde": [], "ic": [], "bc": [], "total": []}

    # 空间网格固定。矩阵读入全部空间点，只输出内部点的二阶导数。
    x_axis = torch.linspace(-1, 1, N)
    h = x_axis[1] - x_axis[0]
    D2 = torch.zeros(N - 2, N, dtype=x_axis.dtype)
    for i in range(N - 2):
        D2[i, i] = 1.0
        D2[i, i + 1] = -2.0
        D2[i, i + 2] = 1.0
    D2 = D2 / h**2
    assert torch.isfinite(D2).all().item(), "D2 contains NaN or Inf"

    print(f"Method: {method}, seed: {seed}, steps: {n_steps}")
    print(f"Input points: {n_times * N}, PDE points: {n_times * (N - 2)}")
    print(f"Device: {next(net.parameters()).device}, dtype: {x_axis.dtype}")
    net.train()
    start_time = perf_counter()

    for step in range(n_steps):
        optimizer.zero_grad()

        # 每组空间点共享一个时间；展开后对每个点的时间输入分别求导。
        times = torch.rand(n_times)
        t = times.repeat_interleave(N).reshape(-1, 1)
        x = x_axis.repeat(n_times).reshape(-1, 1)
        t_ic, x_ic = sample_initial(n_samples)
        t_bc, x_left, x_right = sample_boundary(n_samples)
        t.requires_grad_(True)
        x.requires_grad_(method == "standard")

        u_pred = net(torch.cat((t, x), dim=1))
        u_pred_ic = net(torch.cat((t_ic, x_ic), dim=1))
        u_pred_bl = net(torch.cat((t_bc, x_left), dim=1))
        u_pred_br = net(torch.cat((t_bc, x_right), dim=1))

        # 时间导数始终使用自动求导，并保留计算图以便损失传回网络参数。
        u_t = torch.autograd.grad(
            outputs=u_pred,
            inputs=t,
            grad_outputs=torch.ones_like(u_pred),
            create_graph=True,
        )[0]
        U = u_pred.reshape(n_times, N)
        Ut = u_t.reshape(n_times, N)

        # 两种空间求导方式都只在相同的内部点形成 PDE 残差。
        if method == "standard":
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
            Uxx = u_xx.reshape(n_times, N)[:, 1:-1]
        else:
            Uxx = U @ D2.T

        residual = Ut[:, 1:-1] - alpha * Uxx

        # 方程、初始条件和边界条件共同约束解，避免仅满足 PDE 的零解。
        loss_pde = L_PDE(residual)
        loss_ic = L_IC(u_pred_ic, x_ic)
        loss_bc = L_BC(u_pred_bl, u_pred_br)
        loss_total = L_total(loss_pde, loss_ic, loss_bc)
        if not torch.isfinite(loss_total).item():
            raise FloatingPointError(f"Non-finite loss at step {step + 1}")

        history["pde"].append(loss_pde.item())
        history["ic"].append(loss_ic.item())
        history["bc"].append(loss_bc.item())
        history["total"].append(loss_total.item())

        # 首步确认 PDE 损失能影响网络权重，随后按总损失更新参数。
        if step == 0:
            print("U, Ut:", U.shape, Ut.shape)
            print("Uxx, residual:", Uxx.shape, residual.shape)
            pde_grad = torch.autograd.grad(
                loss_pde, net[0].weight, retain_graph=True
            )[0]
            print("PDE gradient norm:", pde_grad.norm().item())

        loss_total.backward()
        optimizer.step()
        scheduler.step()

        if (step + 1) % 200 == 0 or step + 1 == n_steps:
            print(
                f"Step {step + 1}, PDE: {loss_pde.item():.6f}, "
                f"IC: {loss_ic.item():.6f}, BC: {loss_bc.item():.6f}, "
                f"Total: {loss_total.item():.6f}"
            )
            print("Next learning rate:", scheduler.get_last_lr()[0])

    elapsed = perf_counter() - start_time
    return net, history, elapsed


def evaluate(net, alpha):
    # 用独立的固定时空网格评价解误差；训练与解析解使用同一个 alpha。
    net.eval()
    with torch.no_grad():
        t_axis = torch.linspace(0, 1, 101)
        x_axis = torch.linspace(-1, 1, 201)
        t_grid, x_grid = torch.meshgrid(t_axis, x_axis, indexing="ij")
        t_eval = t_grid.reshape(-1, 1)
        x_eval = x_grid.reshape(-1, 1)
        u_pred = net(torch.cat((t_eval, x_eval), dim=1)).reshape(t_grid.shape)
        u_exact = exact_solution(t_eval, x_eval, alpha).reshape(t_grid.shape)
        error = u_pred - u_exact
        metrics = {
            "MAE": error.abs().mean().item(),
            "relative_L2": torch.sqrt(
                torch.sum(error**2) / torch.sum(u_exact**2)
            ).item(),
            "initial_max_error": error[0].abs().max().item(),
            "left_boundary_max_error": u_pred[:, 0].abs().max().item(),
            "right_boundary_max_error": u_pred[:, -1].abs().max().item(),
        }
    return metrics, x_axis, u_pred, u_exact


def main(output_dir=None, show_plot=True):
    net, history, elapsed = train(
        method=method,
        seed=seed,
        n_steps=n_steps,
        N=N,
        n_times=n_times,
        n_samples=n_samples,
        alpha=alpha,
    )
    metrics, x_axis, u_pred, u_exact = evaluate(net, alpha)
    print(f"Training time: {elapsed:.2f} seconds")
    for name, value in metrics.items():
        print(f"{name}: {value:.8f}")
    print(f"Relative L2 (%): {metrics['relative_L2'] * 100:.2f}%")

    # 文件名区分方法和实验设置，配置、计时与误差一并保存，便于对照。
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = (
        f"compare_demo2_{method}_seed{seed}"
        f"_N{N}_nt{n_times}_steps{n_steps}"
    )
    torch.save(history, output_dir / f"loss_{run_name}.pt")
    torch.save(net.state_dict(), output_dir / f"model_{run_name}.pt")
    report = {
        "config": {
            "method": method,
            "seed": seed,
            "n_steps": n_steps,
            "N": N,
            "n_times": n_times,
            "n_samples": n_samples,
            "alpha": alpha,
            "hidden_layers": 6,
            "hidden_width": 50,
            "activation": "Tanh",
            "device": str(next(net.parameters()).device),
            "dtype": str(next(net.parameters()).dtype),
            "optimizer": "Adam",
            "initial_learning_rate": 0.001,
            "scheduler_step_size": 1000,
            "scheduler_gamma": 0.5,
            "loss_weights": [1.0, 0.5, 0.5],
            "evaluation_shape": [101, 201],
        },
        "training_seconds": elapsed,
        "metrics": metrics,
    }
    report_path = output_dir / f"metrics_{run_name}.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    # 初始切片展示拟合形状，分项损失展示训练过程，全域精度以评估指标为准。
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(x_axis.numpy(), u_pred[0].numpy(), label="Predicted")
    axes[0].plot(x_axis.numpy(), u_exact[0].numpy(), label="Exact")
    axes[0].set(xlabel="x", ylabel="u", title="Solution at t=0.0")
    axes[0].legend()
    for key, label in (("pde", "PDE"), ("ic", "IC"), ("bc", "BC"), ("total", "Total")):
        axes[1].plot(range(1, len(history[key]) + 1), history[key], label=label)
    axes[1].set(xlabel="Step", ylabel="Loss", title="Training Loss", yscale="log")
    axes[1].legend()
    fig.suptitle(f"{method} | seed={seed} | N={N} | time groups={n_times}")
    fig.tight_layout()
    fig.savefig(output_dir / f"plot_{run_name}.png", dpi=200, bbox_inches="tight")
    print("Results saved to:", output_dir)
    print("Metrics file:", report_path)
    if show_plot:
        plt.show()
    plt.close(fig)
    return report


if __name__ == "__main__":
    main()
