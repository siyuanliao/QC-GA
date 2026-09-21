"""运行论文四个声呐案例中的 QC-GA 路径规划实验。"""

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from objective_function import PathObjectiveEvaluator
from QC_GA import QC_GA
from vis_func import vis_func


X_MAX, Y_MAX = 200.0, 100.0
SEARCH_TIME_H = 48.0
SHIP_SPEED_KNOT = 12.0
LEN_LIMIT = SHIP_SPEED_KNOT * 1.852 * SEARCH_TIME_H

N_PATHS = 6
N_D = 150
CON_MIN, CON_MAX = 0.01, 0.99

QCGA_ARGS = dict(
    epsilon=0.7,
    epsilon_min=0.05,
    epsilon_decay=0.97,
    learning_rate=0.1,
    discount_factor=0.9,
    migration_interval=10,
    elite_ratio=0.10,
    migration_ratio=0.10,
    gaussian_sigma0=0.05,
    reward_max=10.0,
    reward_pos=1.0,
    reward_neg=-1.0,
)


class ObjectiveCounter:
    def __init__(self, func):
        self.func = func
        self.count = 0

    def __call__(self, x):
        self.count += 1
        return self.func(x)


def load_case(case_id):
    path = Path("sonar_data") / f"model{case_id}.npy"
    if not path.exists():
        raise FileNotFoundError(f"未找到声呐数据文件: {path}")
    # 数据文件按 (x, y, 方位角) 保存，目标函数内部统一使用 (y, x, 方位角)。
    return np.load(path).transpose(1, 0, 2)


def build_objective(sonar):
    return PathObjectiveEvaluator(
        sonar, X_MAX, Y_MAX,
        len_limit=LEN_LIMIT,
        n_d=N_D,
        d_min=2,
        sublen_min=5,
        no_nav=np.array([50.0, 50.0, 6.0]),
        coverage_nx=100,
        coverage_ny=100,
    )


def normalize_history(history, n):
    arr = np.asarray(history, dtype=float).reshape(-1)
    if arr.size >= n:
        return arr[-n:]
    if arr.size == 0:
        return np.full(n, np.nan)
    return np.r_[arr, np.full(n - arr.size, arr[-1])]


def representative_run(records):
    feasible = [i for i, r in enumerate(records) if r["best_obj"] < 0]
    if not feasible:
        return int(np.argmin([r["best_obj"] for r in records]))
    cov = np.array([-records[i]["best_obj"] for i in feasible])
    med = np.median(cov)
    return feasible[int(np.argmin(np.abs(cov - med)))]


def run_once(objective, pop_size, max_iter, seed):
    rng_init = np.random.default_rng(seed)
    population = rng_init.uniform(
        CON_MIN, CON_MAX, size=(pop_size, N_PATHS + 1)
    )

    initial_obj = np.array([objective(x) for x in population], dtype=float)
    counted_obj = ObjectiveCounter(objective)

    start = time.perf_counter()
    result = QC_GA(
        population,
        counted_obj,
        CON_MIN,
        CON_MAX,
        max_iter=max_iter,
        rng=np.random.default_rng(seed),
        verbose=False,
        **QCGA_ARGS,
    )
    elapsed = time.perf_counter() - start

    best_var, best_obj, current_vars, avg_hist, best_hist = result
    best_hist = np.r_[np.min(initial_obj), normalize_history(best_hist, max_iter)]
    avg_hist = np.r_[np.mean(initial_obj), normalize_history(avg_hist, max_iter)]

    return {
        "seed": seed,
        "best_var": np.asarray(best_var),
        "best_obj": float(best_obj),
        "coverage": -float(best_obj) if best_obj < 0 else np.nan,
        "current_vars": np.asarray(current_vars),
        "history_avg": avg_hist,
        "history_best": best_hist,
        "elapsed_s": elapsed,
        "fes": counted_obj.count,
    }


def save_results(records, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage = np.array([r["coverage"] for r in records], dtype=float)
    valid = coverage[np.isfinite(coverage)]
    times = np.array([r["elapsed_s"] for r in records], dtype=float)
    fes = np.array([r["fes"] for r in records], dtype=float)

    summary = {
        "coverage_mean": np.mean(valid) if valid.size else np.nan,
        "coverage_std": np.std(valid, ddof=1) if valid.size > 1 else np.nan,
        "coverage_variance": np.var(valid, ddof=1) if valid.size > 1 else np.nan,
        "coverage_median": np.median(valid) if valid.size else np.nan,
        "coverage_best": np.max(valid) if valid.size else np.nan,
        "coverage_worst": np.min(valid) if valid.size else np.nan,
        "time_mean_s": np.mean(times),
        "fes_mean": np.mean(fes),
    }

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in summary.items():
            writer.writerow([key, "" if not np.isfinite(value) else f"{value:.3f}"])

    with (output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run", "seed", "coverage", "elapsed_s", "FEs",
            "best_heading_deg", "best_path_positions"
        ])
        for i, r in enumerate(records, 1):
            x = r["best_var"]
            writer.writerow([
                i, r["seed"],
                "" if not np.isfinite(r["coverage"]) else f"{r['coverage']:.3f}",
                f"{r['elapsed_s']:.3f}", r["fes"],
                f"{180.0 * x[0]:.3f}",
                np.array2string(x[1:], precision=3, separator=","),
            ])

    return summary


def plot_convergence(records, max_iter, output_dir):
    steps = np.array([0, 20, 40, 60, 80, 100])
    steps = steps[steps <= max_iter]

    curves = []
    for r in records:
        obj = r["history_best"]
        curves.append(np.where(obj < 0, -obj, np.nan))
    curves = np.asarray(curves)

    mean = np.nanmean(curves, axis=0)
    std = np.nanstd(curves, axis=0, ddof=1) if len(records) > 1 else np.zeros_like(mean)

    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.errorbar(steps, mean[steps], yerr=std[steps], marker="o",
                capsize=3, lw=1.6, label="QC-GA")
    ax.set(xlabel="迭代次数", ylabel="最优搜索覆盖率 / %")
    ax.set_xticks(steps)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "convergence.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="QC-GA 水下声学搜索路径规划实验")
    p.add_argument("--case", type=int, choices=range(1, 5), default=1,
                   help="声呐案例编号：1、2、3 或 4")
    p.add_argument("--runs", type=int, default=1, help="独立重复实验次数")
    p.add_argument("--seed", type=int, default=2026, help="基础随机种子")
    p.add_argument("--pop-size", type=int, default=50)
    p.add_argument("--max-iter", type=int, default=100)
    p.add_argument("--no-vis", action="store_true", help="不绘制代表实验路径")
    return p.parse_args()


def main():
    args = parse_args()
    if args.pop_size % 2:
        raise ValueError("QC-GA 采用等规模双子群，--pop-size 必须为偶数")

    sonar = load_case(args.case)
    evaluator = build_objective(sonar)

    def objective(x):
        return evaluator(x[0], x[1:])

    records = []
    for i in range(args.runs):
        seed = args.seed + i
        record = run_once(objective, args.pop_size, args.max_iter, seed)
        records.append(record)
        cov = record["coverage"]
        cov_text = f"{cov:.3f}%" if np.isfinite(cov) else "无可行解"
        print(f"第 {i+1}/{args.runs} 次：覆盖率={cov_text}，"
              f"耗时={record['elapsed_s']:.3f}s，FEs={record['fes']}")

    out = Path(f"results_case{args.case}")
    summary = save_results(records, out)
    plot_convergence(records, args.max_iter, out)

    rep = records[representative_run(records)]
    x = rep["best_var"]
    print(f"代表实验航向角: {180.0 * x[0]:.3f}°")
    print(f"子路径位置: {np.array2string(x[1:], precision=3)}")
    print(f"平均覆盖率: {summary['coverage_mean']:.3f}%")

    if not args.no_vis:
        vis_func(
            x[0], x[1:], sonar, X_MAX, Y_MAX,
            len_limit=LEN_LIMIT, n_d=N_D,
            no_nav=np.array([50.0, 50.0, 6.0]),
            prefix=str(out / f"case{args.case}_QC_GA"),
        )
        plt.show()


if __name__ == "__main__":
    main()
