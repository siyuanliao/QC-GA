"""QC-GA 优化结果的可视化辅助函数。"""

import numpy as np
from objective_function import PathObjectiveEvaluator
from plot_road import vis_road
from plot_sonar_search import plot_sonar_search


def vis_func(heading_angle, path_positions, sonar_matrix, x_max, y_max,
             len_limit, n_d=150, no_nav=np.array([50.0, 50.0, 6.0]),
             prefix="case1_QC_GA"):
    evaluator = PathObjectiveEvaluator(
        sonar_matrix, x_max, y_max, len_limit=len_limit, n_d=n_d,
        d_min=0.0, sublen_min=0.0, no_nav=no_nav,
        coverage_nx=100, coverage_ny=100
    )
    details = evaluator.evaluate_details(
        heading_angle, path_positions, enforce_constraints=False
    )

    print(f"路径总长度: {details['total_length']:.2f} km")
    print(f"搜索覆盖率: {details['coverage_percent']:.2f}%")

    vis_road(details["roads"], x_max, y_max)
    plot_sonar_search(
        details["roads"], details["mpx"], details["mpy"],
        details["coverage_mask"], no_nav, save_path=f"{prefix}_cover_path.png"
    )
    return details
