# 一维热传导 PINN 复现与对照实验 - Design Spec

## I. Project Information
- Project Name: 一维热传导 PINN 复现与对照实验
- Page Count: 6 total, approximately 6 minutes (用户要求增加实际结果图)
- Target Audience: 导师与课题组
- Use Case: 组会项目进展汇报
- Created Date: 2026-10-28
- Approval: 用户接受内容方案，并要求仅使用所提供 PPTX 的背景模板。

## II. Canvas Specification
- Dimensions: 960 × 544.333333 px
- Native size: 9144000 × 5184775 EMU, matching source exactly
- Content margins: 70 px left/right, 42 px top, 30 px bottom

## III. Visual Theme
- Reference: C:/Users/Kendrick何/OneDrive/桌面/PINN前沿/ppt-2/PINN 发展脉络与最新进展2026-10-15.pptx
- Background: 原始封面建筑背景 image1.jpg 与校徽 image2.png。内页与原模板一致，使用白底。
- Primary: #A51E36 暗红
- Body: #555759 灰色
- Secondary: #6C6E70
- Divider: #D8D8D8
- 不沿用参考演示的正文、日期、文献数据或内容布局。

## IV. Typography System
- Typography direction: 保留源文件等线字体 DengXian
- Title / Body / Emphasis: DengXian
- Mathematical characters: Cambria Math as needed
- Title: 32 px; cover: 38 px; body: 22–24 px; annotation: 16–18 px; footnote: 13–14 px
- 原模板中等线已核实。公式如需 Cambria Math，属于新增数学排版字体。

## V. Layout Principles
- 以单一整洁画面组织每页，不套用源内容结构。
- P01: 封面背景上的标题及数学问题摘要，不另加纯封面。
- P02: 输入、网络、输出流程，下方并列求导方法与共用损失。
- P03: 三组原问题与对应修正，突出同一时刻空间采样。
- P04: 两个等刻度柱状图，分别比较求导方法和时间采样。
- P05: 三个时刻的预测与解析解曲线，5000 步训练总损失。
- P06: 当前精度与完成范围，接后续验证和优化工作。

## VI. Icon Usage Specification
- No decorative icons. Preserve source ECNU image unchanged.

## VII. Visualization Reference List
- Catalog read: 70 templates / 10 categories
- P02: 自定义可编辑计算流程，无完整匹配模板。
- P04: 两个原生可编辑柱状图，均显示整体相对 L2 误差，统一 0–15% 纵轴。
- Runners-up: grouped_bar_chart（会将两种不同实验误认为一个共同因子）；dumbbell_chart（两组小样本用柱状图更直接）；process_flow（通用流程缺少本项目的两条求导支路）。
- Native chart generation takes precedence over SVG chart export to preserve editable chart data.

## VIII. Image Resource List
| Filename | Dimensions | Purpose | Acquire Via | Status |
|---|---|---|---|---|
| images/image1.jpg | 2418×1371 | 原始封面背景，完整保留 | user | Existing |
| images/image2.png | 2187×706 | 原始校徽，保持长宽比 | user | Existing |

## IX. Content Outline
### Slide 01 — 一维热传导 PINN 复现
- 方程 u_t = αu_xx；α=1；x∈[-1,1]，t∈[0,1]
- 初值 sin(πx)，两端零边界，解析解 exp(-απ²t)sin(πx)
- 解析解用于独立验证，不作为内部监督数据。
### Slide 02 — 标准 PINN 与混合 PINN
- 共用 6×50 Tanh 网络、Adam、5000 步。
- 标准：时间和空间均自动求导。
- 混合：时间自动求导，空间三点二阶差分。
- 共同损失 PDE + 0.5 IC + 0.5 BC。
### Slide 03 — 原仓库的关键修正
- 正确解析解及一致的 α。
- 同一时间的空间点才能构成空间差分。
- 网格和矩阵一起传递，只对内部点计算空间残差。
### Slide 04 — 方法对照与时间采样实验
- 标准分组 11.237432%，混合分组 11.031993%。
- 标准分组 11.237432%，标准逐点独立时间 6.203807%。
- seed42、CPU、5000 步，固定 101×201 评估网格。
- 计时 42.28 s / 26.10 s 只作本次观察。混合耗时由终端截图转录。
- 单种子结果不能说明普遍优劣。
### Slide 05 — 温度预测与训练过程
- 标准 PINN、逐点独立时间、seed42、5000 步。
- t=0、0.1、0.3 的预测与解析解，颜色对应时间，实线解析、虚线预测。
- 完整原始 5000 步 log10 总损失，无平滑、无抽样。
- 保存权重重新评估与原记录一致，相对 L2=6.203807%。
### Slide 06 — 当前进度与后续工作
- 基础复现、独立验证及采样对照已完成。
- 逐点独立时间的标准方法：相对 L2 为 6.20%，仍未达到 5%。
- 初值最大误差 0.0281、边界最大误差 0.0496，均达到 0.05 目标。
- 下一步：seed43 与重复计时，增加时间覆盖，评估 Adam 后接 L-BFGS。

## X. Speaker Notes Requirements
- 每页约 45–75 秒中文讲稿，嵌入 PPT 备注。
- 引用代码及 outputs/results.json；明确单种子与混合计时来源。
- 讲稿可以解释参数与公式，正文不堆代码。

## XI. Technical Constraints Reminder
- 使用 Artifact Tool 原生文本、连接线和数据图表制作并导出 PPTX。
- 原始图片字节保持不变，不修改参考 PPTX。
- 现有五页保留，增加一页结果图；检查所有页面。
- 输出至 exports/；构建中间文件留在 build/。
