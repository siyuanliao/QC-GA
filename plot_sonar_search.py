"""绘制搜索覆盖区域、航行路径和禁航区。"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, Patch, Rectangle
from matplotlib.lines import Line2D


def plot_sonar_search(roads, mpx, mpy, cover_count, no_nav, save_path=None):
    roads = np.asarray(roads, dtype=float)
    cover = (np.asarray(cover_count) > 0).astype(int)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    cmap = ListedColormap(["#ECECEC", "#B7D9EE"])
    ax.pcolormesh(mpx, mpy, cover, cmap=cmap, shading="nearest", zorder=1)
    ax.contour(mpx, mpy, cover, levels=[0.5], colors="#2B6F9C",
               linestyles="--", linewidths=1.2, zorder=2)

    xmin, xmax = float(np.min(mpx)), float(np.max(mpx))
    ymin, ymax = float(np.min(mpy)), float(np.max(mpy))
    ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                           fill=False, ls="--", lw=1.2, color="black"))

    if no_nav is not None and len(no_nav) >= 3 and no_nav[2] > 0:
        ax.add_patch(Circle((no_nav[0], no_nav[1]), no_nav[2],
                            facecolor="#F4B7B2", edgecolor="#B6403A", lw=1.2, zorder=3))

    ax.plot(roads[:, 0], roads[:, 1], "-o", color="#263746", lw=1.4, ms=3, zorder=4)
    ax.scatter(*roads[0], marker="s", s=45, color="#2CA25F", zorder=5)
    ax.scatter(*roads[-1], marker="*", s=85, color="#3568B8", zorder=5)

    handles = [
        Patch(facecolor="#B7D9EE", label="已覆盖区域"),
        Patch(facecolor="#ECECEC", label="未覆盖区域"),
        Patch(facecolor="#F4B7B2", edgecolor="#B6403A", label="禁航区"),
        Line2D([0], [0], color="#263746", marker="o", lw=1.4, label="航行路径"),
    ]
    ax.legend(handles=handles, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.set(xlabel="X 坐标 (km)", ylabel="Y 坐标 (km)", xlim=(xmin, xmax), ylim=(ymin, ymax))
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax
