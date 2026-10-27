# 已保存的实验结果

本目录包含四组种子 42、5000 步训练的模型与损失历史。发布整理不重新训练，也不覆盖历史权重。

在项目根目录运行 `python evaluate_saved.py`，可重新加载四份权重，使用同一个解析解和固定网格验证五项误差、损失条数、有限性及加权关系。脚本只读取文件。

| 实验标识 | 模型文件 | 损失历史 |
| --- | --- | --- |
| `historical` | [原标准模型](model_seed42_steps5000.pt) | [历史](loss_seed42_steps5000.pt) |
| `standard_grouped` | [分组标准模型](model_compare_demo2_standard_seed42_N50_nt4_steps5000.pt) | [历史](loss_compare_demo2_standard_seed42_N50_nt4_steps5000.pt) |
| `hybrid_grouped` | [分组混合模型](hybrid_model_seed42_steps5000.pt) | [历史](hybrid_loss_seed42_steps5000.pt) |
| `standard_independent` | [独立时间标准模型](model_time_sampling_demo3_independent_seed42_N50_copies4_steps5000.pt) | [历史](loss_time_sampling_demo3_independent_seed42_N50_copies4_steps5000.pt) |

[results.json](results.json) 汇总共同配置、文件对应关系、误差和计时来源。`relative_L2` 保存小数，例如 `0.062038` 表示约 `6.2038%`。

两份 `metrics_*.json` 是训练时生成的原始记录；混合方法的 26.10 秒来自用户终端截图转录，截图未包含在仓库中，历史标准方法未记录耗时。加载模型只能复核误差，不能恢复当时的训练耗时。

模型文件保存 `state_dict`，不含优化器状态，因此支持评估，不支持严格恢复当时的训练进度。历史使用 `torch.save` 保存字典。加载时使用 `weights_only=True`。

图像：

- [原标准训练损失](loss_seed42_steps5000.png)
- [分组标准：初始切片与损失](plot_compare_demo2_standard_seed42_N50_nt4_steps5000.png)
- [独立时间标准：初始切片与损失](plot_time_sampling_demo3_independent_seed42_N50_copies4_steps5000.png)

图中的解曲线仅是 `t=0` 切片；全时空精度应查看 `relative_L2` 等评估指标。
