# MacBook 接续说明

## 已保存的工作

- `train_compare_demo2.py`：训练日志间隔改为 500 步。
- `train_hybrid_demo.py`：随机种子改为变量，当前为 43；输出文件名随种子变化。
- `outputs/*compare_demo2_standard_seed42_N50_nt4_steps10000*`：新增 10000 步标准方法实验的模型、损失、指标和图。
- `refine_lbfgs.py`：加载已有模型并评估精调前误差；尚未实现 L-BFGS 精调。
- `rewrite.py`：学习过程中的重写草稿，保留原样；不作为验证通过的实验入口。
- `projects/pinn_heat_group_meeting_ppt169_20261028/`：汇报 PPT、素材、讲稿、制作脚本和预览。

原有发布结果仍见 `README.md` 和 `docs/EXPERIMENTS.md`。本次保存工作进展，没有重新运行训练，也没有将新增实验混入原发布结果表。

## 在 Mac 上取得项目

1. 登录拥有该 GitHub 仓库访问权限的账号，在仓库页面选择 **Code**，复制仓库地址；用 Git 客户端克隆，或在终端执行 `git clone <仓库地址>`。
2. 如 Mac 已有该仓库，先确认没有需要保留的本地改动，再切到 `master`，执行 `git pull --ff-only origin master`。
3. 在 Mac 的 Codex 中将这个本地文件夹添加为项目。聊天移交与本地文件可用性应分别确认。
4. 保持 Mac 的 Codex 打开，并按应用内 Hand off 菜单选择 MacBook；如果列表中没有 MacBook，先检查目标设备的登录和连接状态。

## 仍在 Git 之外的文件

以下内容保留原有忽略规则，不随 GitHub 克隆下载。如需继续使用，单独复制并保持相对路径：

- `1.py`
- `train_pinn_demo.py`
- `outputs/hybrid_model_seed42_steps100.pt`
- `outputs/hybrid_loss_seed42_steps100.pt`
- `.vscode/settings.json`（可选；其中的本机路径需要在 Mac 上重新选择）

`__pycache__/`、`.chart-data-*/` 和 `node.*.log` 是缓存或中间文件，通常无需迁移。

PPT 原始参考模板位于项目外的 `PINN前沿/ppt-2/PINN 发展脉络与最新进展2026-10-15.pptx`。若需要继续使用完整原始模板，请另外复制。汇报已经提取的图片和导出的 PPT 在本仓库内。

## 环境与验证

Windows 的 Python/Conda 环境不能直接作为 Mac 环境使用。请在 Mac 重新创建环境；已有的 `environment.yml` 和 `requirements.txt` 记录了 Windows 实验使用的 Python 及直接依赖版本，尚未在 Mac 验证安装。

Mac 使用适配其系统和芯片的 PyTorch 安装方式，不使用 README 中的 NVIDIA CUDA 安装命令。若固定版本在 Mac 无法安装，先记录具体报错并核对可用版本，不要直接改写基线结果。

环境就绪后，在项目根目录运行：

```sh
python check_env.py
python check_pde.py
python check_spatial.py
python check_hybrid.py
python evaluate_saved.py
```

这些检查不要求重新训练。新平台的浮点误差和运行耗时可能不同。

PPT 的 `build/finalize*.mjs` 引用了 Windows 的 Codex 工具和解释器绝对路径。已导出的 PPT 可以直接使用；重新制作前需重新定位 Mac 上的对应工具，不能直接沿用 Windows 路径。
