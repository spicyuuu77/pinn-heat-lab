import torch
import matplotlib.pyplot as plt
from pathlib import Path

if __name__ == "__main__":

    history_path = Path(__file__).resolve().parent / "outputs" / "loss_seed42_steps5000.pt"
    history = torch.load(history_path, weights_only=True)

    pde_values = history["pde"]  # 取出损失列表
    steps = range(1, len(pde_values) + 1)  # 配上第 1～5000 步

    IC_values = history["ic"]
    IC_steps = range(1, len(IC_values) + 1)

    BC_values = history["bc"]
    BC_steps = range(1, len(BC_values) + 1)


    plt.plot(steps, pde_values, label="PDE")  # 画线
    plt.plot(IC_steps, IC_values, label="IC")
    plt.plot(BC_steps, BC_values, label="BC")
    plt.plot(steps, history["total"], label="Total")
    plt.yscale("log")  # 设置纵轴为对数坐标
    plt.xlabel("Step")  # 给横轴写名字
    plt.ylabel("Loss")  # 给纵轴写名字
    plt.legend()  # 显示标有“PDE”的图例
    figure_path = history_path.with_suffix(".png")
    plt.savefig(figure_path, dpi=200, bbox_inches="tight")
    print("Figure saved to:", figure_path)
    plt.show()  # 显示图像窗口
