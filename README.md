# 一维热传导方程的 PINN 学习项目

本项目用于从基础开始学习物理信息神经网络（PINN），依次独立实现标准 PINN 与混合求导 PINN，并通过解析解检查计算结果。同时练习环境管理、代码组织、实验记录和 Git 版本管理。

## 项目目标

参考仓库：[PINN solves the one-dimensional heat conduction equation](https://github.com/yydjya/PINN-solves-the-one-dimensional-heat-conduction-equation)。

求解以下一维热传导问题，第一轮固定热扩散系数为 `α = 1`：

$$
\frac{\partial u}{\partial t}=\alpha\frac{\partial^2u}{\partial x^2},\qquad x\in[-1,1],\quad t\in[0,1]
$$

初始条件与边界条件为：

$$
u(x,0)=\sin(\pi x),\qquad u(-1,t)=u(1,t)=0
$$

用于验证结果的解析解为：

$$
u(x,t)=e^{-\alpha\pi^2t}\sin(\pi x)
$$

计划先使用自动求导完成标准 PINN，再实现时间自动求导与空间局部多项式求导相结合的混合 PINN。最终在相同问题和可比较的训练设置下，记录两种方法的误差与耗时。

参考代码中已发现解析解、评估参数传递和混合求导实现等问题，复现时需要逐项修正。正确性以方程、初边值条件和经过验证的解析解为依据。

## 运行环境

以下为 2026-10-22 已验证的环境：

| 项目 | 配置 |
| --- | --- |
| 操作系统 | Windows |
| 编辑器 | VS Code |
| Conda 环境名 | `pinn-heat-lab` |
| Python | `3.10.10` |
| PyTorch | `2.14.1+cu126` |
| PyTorch CUDA 运行库 | `12.6` |
| NumPy | `1.23.5` |
| Matplotlib | `3.7.2` |
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop GPU，4 GB 显存 |

本机解释器位置：`D:\anaconda\envs\pinn-heat-lab\python.exe`。

PyTorch 使用[官方 CUDA 12.6 软件包源](https://download.pytorch.org/whl/cu126)。项目代码与 Conda 环境分别存放，运行时通过选择解释器建立对应关系。

本节记录当前验证过的主要依赖；完整的环境重建配置尚待补充。

## 运行方法

在 VS Code 中打开本项目文件夹，通过 `Python: Select Interpreter` 选择 `pinn-heat-lab` 对应的解释器。

在已经完成 Conda 初始化的 PowerShell 中，切换到本项目根目录，逐条执行：

```powershell
conda activate pinn-heat-lab
python check_env.py
```

当前脚本检查 PyTorch 版本、CUDA 可用状态和 Python 解释器路径，输出应包含：

| 检查项 | 本机预期结果 |
| --- | --- |
| PyTorch 版本 | `2.14.1+cu126` |
| CUDA 可用状态 | `True` |
| Python 解释器路径 | `D:\anaconda\envs\pinn-heat-lab\python.exe` |

确认运行环境时，以实际解释器路径为准。不同环境可能使用相同的 Python 版本。

## 当前进度

- [x] 完成 Conda 环境激活与 Python 解释器检查。
- [x] 安装 GPU 版 PyTorch，并验证基本 GPU 张量计算及 NumPy 数据转换。
- [x] 编写并运行环境检查脚本 `check_env.py`。
- [x] 建立初步项目说明和 `.gitignore`。
- [ ] 建立 Git 版本记录，补充环境重建配置。
- [ ] 验证解析解，完成张量与自动求导练习。
- [ ] 独立实现并验证标准 PINN。
- [ ] 独立实现并验证混合 PINN。
- [ ] 绘制验证图，记录误差、耗时及实验结论。

目前完成的是环境准备与检查，尚未实现 PINN 求解器，也尚未完成训练或精度验收。
