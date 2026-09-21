# QC-GA 水下声学搜索路径规划代码

本仓库提供论文中角度自适应水下声学搜索路径规划实验的主要代码，包括 QC-GA 优化算法、目标函数、路径与覆盖区域可视化，以及 4 种对比算法。

## 1. 目录结构

```text
.
├── main_apls.py              # QC-GA 主程序
├── QC_GA.py                  # 本文提出的 QC-GA 算法
├── objective_function.py     # 路径构造、约束和覆盖率目标函数
├── vis_func.py               # 优化结果可视化入口
├── plot_road.py              # 搜索路径绘图
├── plot_sonar_search.py      # 搜索覆盖区域绘图
├── DE_NPS.py                 # DE-NPS 对比算法
├── KPMBO.py                  # KPMBO 对比算法
├── I_LNS.py                  # I-LNS 对比算法
├── MACPSO.py                 # MACPSO 对比算法
├── requirements.txt          # Python 依赖
└── sonar_data/
    ├── model1.npy
    ├── model2.npy
    ├── model3.npy
    └── model4.npy
```

`model1.npy` 至 `model4.npy` 分别对应论文中的案例 1~4。

## 2. 运行环境

建议使用 Python 3.10 或更高版本。

```bash
pip install -r requirements.txt
```

其中 `MACPSO.py` 需要 PyTorch。若只运行 QC-GA，主要依赖为 NumPy 和 Matplotlib。

## 3. 运行 QC-GA

运行案例 1：

```bash
python main_apls.py --case 1
```

案例 2~4 的调用方式相同：

```bash
python main_apls.py --case 2
python main_apls.py --case 3
python main_apls.py --case 4
```

主程序默认设置为：

- 种群规模：50；
- 平行搜索子路径数：6；
- 最大迭代次数：100；
- 舰船航速：12 kn；
- 搜索时间：48 h；
- 路径离散采样点数：150；
- 决策变量范围：[0.01, 0.99]。

每个个体的第 1 个变量为归一化航向角，`[0, 1]` 对应实际的 `0~180°`；其余 6 个变量表示平行子路径的位置。

## 4. 声呐数据说明

主程序读取 `sonar_data/modelX.npy`。原始数据文件按 `(x, y, 方位角)` 保存，读取后会交换前两个维度，因此目标函数内部统一采用：

```text
(y_index, x_index, bearing_index)
```

第三维为绝对地理方位角，0° 表示正北，角度沿顺时针方向增加。

## 5. 输出结果

程序运行后会在 `results_caseX/` 中保存结果，例如：

```text
summary.csv         # 多次实验的覆盖率统计、平均耗时和平均 FEs
runs.csv            # 每次独立实验的结果
convergence.png     # QC-GA 收敛曲线
caseX_QC_GA_cover_path.png   # 代表实验的搜索路径与覆盖区域
```

当进行多次独立实验时，程序选择最终覆盖率最接近中位数的一次作为代表实验，用于路径可视化。

## 6. 对比算法

`DE_NPS.py`、`KPMBO.py`、`I_LNS.py` 和 `MACPSO.py` 保留了论文对比实验中使用的实现形式，调用接口与 QC-GA 基本一致：

```python
best_x, best_f, population, history_avg, history_best = optimizer(
    initial_population,
    objective,
    lower_bound,
    upper_bound,
    max_iter=100,
)
```

需要说明：

- `DE_NPS.py` 为本文对比实验使用的 DE-NPS 实现；
- `KPMBO.py` 采用批量真实评价，使目标函数评价预算与种群式算法处于相近量级；
- `I_LNS.py` 将大邻域搜索中的破坏—修复思想适配到本文连续变量路径规划问题；
- `MACPSO.py` 使用多 Actor、单 Critic 的粒子群结构，需要 PyTorch。

其中 KPMBO 和 I-LNS 为针对本文优化接口所做的适配版本，目的是复现实验中的对比设置，并不作为对应原始论文在所有问题上的通用参考实现。

## 7. 随机种子

主程序默认基础随机种子为 2026。不同重复实验依次使用不同子种子。可以通过 `--seed` 指定新的基础随机种子：

```bash
python main_apls.py --case 1 --runs 10 --seed 2026
```

在相同代码、数据、参数和软件环境下，固定随机种子便于复现实验结果。不同硬件和软件版本可能导致运行时间存在差异。

## 8. 引用

如果本代码对您的研究有帮助，请引用与本仓库对应的论文：

《面向水下声学搜索的角度自适应路径规划方法》
