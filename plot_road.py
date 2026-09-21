"""绘制搜索路径及航行方向。"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


def vis_road(roads, x_max, y_max):
    roads = np.asarray(roads, dtype=float)
    if roads.ndim != 2 or roads.shape[0] < 2 or roads.shape[1] != 2:
        raise ValueError("roads 必须为 (n, 2) 数组，且 n >= 2")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.add_patch(Rectangle((0, 0), x_max, y_max, fill=False, lw=1.5))

    ax.plot(roads[:, 0], roads[:, 1], "o", ms=4, color="#2C3E50")
    for a, b in zip(roads[:-1], roads[1:]):
        ax.plot([a[0], b[0]], [a[1], b[1]], lw=1.3, color="#2C3E50")
        mid = (a + b) / 2
        vec = b - a
        length = np.linalg.norm(vec)
        if length > 0:
            unit = vec / length
            d = min(0.12 * length, 3.0)
            ax.add_patch(FancyArrowPatch(
                mid - unit * d, mid + unit * d,
                arrowstyle="-|>", mutation_scale=12,
                lw=1.1, color="#2C3E50"
            ))

    ax.scatter(*roads[0], marker="s", s=55, color="#2CA25F", label="起点", zorder=4)
    ax.scatter(*roads[-1], marker="*", s=90, color="#3568B8", label="终点", zorder=4)
    ax.set(xlim=(0, x_max), ylim=(0, y_max), xlabel="X 坐标 (km)", ylabel="Y 坐标 (km)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    return fig, ax
