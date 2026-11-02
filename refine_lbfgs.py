from pathlib import Path

import torch

from evaluate_saved import build_network
from train_compare_demo2 import evaluate


def main():
    alpha = 1.0

    # 使用当前脚本所在目录定位已有模型。
    root = Path(__file__).resolve().parent
    model_path = (
        root
        / "outputs"
        / ("model_time_sampling_demo3_independent" "_seed42_N50_copies4_steps5000.pt")
    )

    # 先创建匹配的网络结构，再恢复已经训练好的参数。
    net = build_network(hidden_layers=6, hidden_width=50)

    # TODO 1：读取权重文件，结果命名为 state_dict。

    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)


    # TODO 2：将 state_dict 装入 net，要求参数严格匹配。

    net.load_state_dict(state_dict)

    # 在精调之前记录误差，作为本轮实验的起点。
    before, _, _, _ = evaluate(net, alpha)

    print("Before L-BFGS:")
    for name, value in before.items():
        print(f"{name}: {value:.8f}")

    print(f"Relative L2: {before['relative_L2'] * 100:.4f}%")


if __name__ == "__main__":
    main()
