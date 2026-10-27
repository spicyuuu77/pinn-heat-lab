import argparse
import json
from pathlib import Path

import torch

from train_compare_demo2 import evaluate


ROOT = Path(__file__).resolve().parent


def build_network(hidden_layers, hidden_width):
    layers = []
    input_width = 2
    for _ in range(hidden_layers):
        layers.extend((torch.nn.Linear(input_width, hidden_width), torch.nn.Tanh()))
        input_width = hidden_width
    layers.append(torch.nn.Linear(input_width, 1))
    return torch.nn.Sequential(*layers).to(device="cpu", dtype=torch.float32)


def check_run(run, config):
    # 从已保存的权重重建网络，只评估，不调用训练或覆盖实验文件。
    net = build_network(config["hidden_layers"], config["hidden_width"])
    state = torch.load(ROOT / run["model"], map_location="cpu", weights_only=True)
    net.load_state_dict(state, strict=True)
    if not all(torch.isfinite(value).all().item() for value in state.values()):
        raise ValueError(f"Non-finite model weights: {run['id']}")
    metrics, _, _, _ = evaluate(net, config["alpha"])

    # 固定网格重新算出的五项误差必须与发布记录一致，允许浮点计算的小差异。
    for key, expected in run["metrics"].items():
        torch.testing.assert_close(
            torch.tensor(metrics[key], dtype=torch.float64),
            torch.tensor(expected, dtype=torch.float64),
            rtol=1e-5,
            atol=1e-6,
            msg=f"Metric mismatch: {run['id']} / {key}",
        )

    # 损失历史也要完整、有限，且总损失与三个分项的权重关系一致。
    history = torch.load(ROOT / run["history"], map_location="cpu", weights_only=True)
    losses = {}
    for key in ("pde", "ic", "bc", "total"):
        values = torch.as_tensor(history[key], dtype=torch.float64)
        if values.shape != (config["n_steps"],) or not torch.isfinite(values).all().item():
            raise ValueError(f"Invalid loss history: {run['id']} / {key}")
        losses[key] = values
    torch.testing.assert_close(
        losses["total"], losses["pde"] + 0.5 * losses["ic"] + 0.5 * losses["bc"],
        rtol=1e-5, atol=1e-7,
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate saved PINN models without training.")
    parser.add_argument("--run", default="all", help="all, historical, standard_grouped, hybrid_grouped, standard_independent")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "outputs/results.json").read_text(encoding="utf-8"))
    selected = [run for run in manifest["runs"] if args.run in ("all", run["id"])]
    if not selected:
        parser.error(f"Unknown run: {args.run}")
    for run in selected:
        metrics = check_run(run, manifest["common_config"])
        print(run["id"])
        for name, value in metrics.items():
            print(f"  {name}: {value:.8f}")
        print(f"  Relative L2: {metrics['relative_L2'] * 100:.4f}%")
    print(f"PASS: {len(selected)} saved runs verified; no training performed.")


if __name__ == "__main__":
    main()
