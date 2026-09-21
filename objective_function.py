"""角度自适应平行线搜索的目标函数。

航向角归一化到 [0, 1]，对应实际的 0~180°；声呐数据按 (y, x, 绝对方位角) 使用。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
_TWO_PI = 2.0 * np.pi
_EPS = 1e-12

@dataclass(frozen=True)
class PathGeometry:
    roads: np.ndarray
    ordered_positions: np.ndarray
    line_offsets: np.ndarray
    subpath_lengths: np.ndarray
    spacings: np.ndarray

def _unit_vectors(theta: float) -> tuple[np.ndarray, np.ndarray]:
    d = np.array([np.sin(theta), np.cos(theta)], dtype=float)
    n = np.array([-np.cos(theta), np.sin(theta)], dtype=float)
    return (d, n)

def _clip_parallel_lines_to_rectangle(offsets: np.ndarray, d: np.ndarray, n: np.ndarray, x_max: float, y_max: float) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.asarray(offsets, dtype=float)
    p0 = offsets[:, None] * n[None, :]
    t_low = np.full(offsets.size, -np.inf, dtype=float)
    t_high = np.full(offsets.size, np.inf, dtype=float)
    for axis, upper in ((0, x_max), (1, y_max)):
        da = d[axis]
        coord0 = p0[:, axis]
        if abs(da) < _EPS:
            outside = (coord0 < -1e-10) | (coord0 > upper + 1e-10)
            if np.any(outside):
                raise ValueError('存在平行直线未与矩形任务区域相交。')
            continue
        t0 = (0.0 - coord0) / da
        t1 = (upper - coord0) / da
        t_axis_low = np.minimum(t0, t1)
        t_axis_high = np.maximum(t0, t1)
        t_low = np.maximum(t_low, t_axis_low)
        t_high = np.minimum(t_high, t_axis_high)
    if np.any(t_low > t_high + 1e-10):
        raise ValueError('存在平行直线未与矩形任务区域相交。')
    starts = p0 + t_low[:, None] * d[None, :]
    ends = p0 + t_high[:, None] * d[None, :]
    starts[:, 0] = np.clip(starts[:, 0], 0.0, x_max)
    starts[:, 1] = np.clip(starts[:, 1], 0.0, y_max)
    ends[:, 0] = np.clip(ends[:, 0], 0.0, x_max)
    ends[:, 1] = np.clip(ends[:, 1], 0.0, y_max)
    return (starts, ends)

def build_search_path(heading_angle: float, path_positions: np.ndarray, x_max: float, y_max: float) -> PathGeometry:
    if not 0.0 <= float(heading_angle) <= 1.0:
        raise ValueError('heading_angle 必须归一化到 [0, 1]。')
    if x_max <= 0 or y_max <= 0:
        raise ValueError('x_max 和 y_max 必须为正数。')
    positions = np.asarray(path_positions, dtype=float).reshape(-1)
    if positions.size == 0:
        raise ValueError('path_positions 不能为空。')
    if np.any(~np.isfinite(positions)) or np.any((positions < 0.0) | (positions > 1.0)):
        raise ValueError('path_positions 中的元素必须为 [0, 1] 范围内的有限数值。')
    theta = float(heading_angle) * np.pi
    d, n = _unit_vectors(theta)
    vertices = np.array([[0.0, 0.0], [x_max, 0.0], [0.0, y_max], [x_max, y_max]], dtype=float)
    projections = vertices @ n
    c_min = projections.min()
    c_max = projections.max()
    offsets = c_min + positions * (c_max - c_min)
    order = np.argsort(-offsets) if theta <= np.pi / 2.0 else np.argsort(offsets)
    offsets = offsets[order]
    ordered_positions = positions[order]
    starts, ends = _clip_parallel_lines_to_rectangle(offsets, d, n, x_max, y_max)
    subpath_lengths = np.linalg.norm(ends - starts, axis=1)
    spacings = np.abs(np.diff(offsets))
    roads = [np.array([0.0, 0.0], dtype=float)]
    for i, (a, b) in enumerate(zip(starts, ends)):
        if i % 2 == 0:
            roads.extend((a, b))
        else:
            roads.extend((b, a))
    return PathGeometry(roads=np.asarray(roads, dtype=float), ordered_positions=ordered_positions, line_offsets=offsets, subpath_lengths=subpath_lengths, spacings=spacings)

def sample_polyline(roads: np.ndarray, n_samples: int) -> tuple[np.ndarray, np.ndarray, float]:
    roads = np.asarray(roads, dtype=float)
    if roads.ndim != 2 or roads.shape[1] != 2 or len(roads) < 2:
        raise ValueError('roads 必须是形状为 (m, 2) 且 m>=2 的数组。')
    if n_samples < 2:
        raise ValueError('n_samples 至少应为 2。')
    vectors_all = roads[1:] - roads[:-1]
    lengths_all = np.linalg.norm(vectors_all, axis=1)
    valid = lengths_all > _EPS
    if not np.any(valid):
        points = np.repeat(roads[:1], n_samples, axis=0)
        return (points, np.zeros(n_samples), 0.0)
    starts = roads[:-1][valid]
    vectors = vectors_all[valid]
    lengths = lengths_all[valid]
    cumulative_end = np.cumsum(lengths)
    cumulative_start = np.concatenate(([0.0], cumulative_end[:-1]))
    total_length = float(cumulative_end[-1])
    s = np.linspace(0.0, total_length, n_samples)
    seg_idx = np.searchsorted(cumulative_end, s, side='right')
    seg_idx = np.clip(seg_idx, 0, len(lengths) - 1)
    local_t = (s - cumulative_start[seg_idx]) / lengths[seg_idx]
    points = starts[seg_idx] + local_t[:, None] * vectors[seg_idx]
    seg_vec = vectors[seg_idx]
    headings = np.mod(np.arctan2(seg_vec[:, 0], seg_vec[:, 1]), _TWO_PI)
    return (points, headings, total_length)

def bilinear_sonar_interpolation(sample_points: np.ndarray, sonar_matrix: np.ndarray, x_max: float, y_max: float) -> np.ndarray:
    sonar = np.asarray(sonar_matrix, dtype=float)
    if sonar.ndim != 3:
        raise ValueError('sonar_matrix 必须具有形状 (ny, nx, n_angle)。')
    ny, nx, _ = sonar.shape
    if nx < 2 or ny < 2:
        raise ValueError('sonar_matrix 在 x、y 两个空间方向上至少都需要 2 个网格点。')
    p = np.asarray(sample_points, dtype=float)
    x = np.clip(p[:, 0], 0.0, x_max) / x_max * (nx - 1)
    y = np.clip(p[:, 1], 0.0, y_max) / y_max * (ny - 1)
    x0 = np.floor(x).astype(np.intp)
    y0 = np.floor(y).astype(np.intp)
    x1 = np.minimum(x0 + 1, nx - 1)
    y1 = np.minimum(y0 + 1, ny - 1)
    wx = (x - x0)[:, None]
    wy = (y - y0)[:, None]
    r00 = sonar[y0, x0, :]
    r10 = sonar[y0, x1, :]
    r01 = sonar[y1, x0, :]
    r11 = sonar[y1, x1, :]
    return (1.0 - wx) * (1.0 - wy) * r00 + wx * (1.0 - wy) * r10 + (1.0 - wx) * wy * r01 + wx * wy * r11

def calculate_coverage_mask(sample_points: np.ndarray, sample_ranges: np.ndarray, grid_x_flat: np.ndarray, grid_y_flat: np.ndarray) -> np.ndarray:
    points = np.asarray(sample_points, dtype=float)
    ranges = np.asarray(sample_ranges, dtype=float)
    gx = np.asarray(grid_x_flat, dtype=float).reshape(-1)
    gy = np.asarray(grid_y_flat, dtype=float).reshape(-1)
    if ranges.ndim != 2 or ranges.shape[0] != len(points):
        raise ValueError('sample_ranges 必须具有形状 (n_samples, n_angle)。')
    n_angle = ranges.shape[1]
    angle_scale = n_angle / _TWO_PI
    covered = np.zeros(gx.size, dtype=bool)
    for point, radial in zip(points, ranges):
        remaining = np.flatnonzero(~covered)
        if remaining.size == 0:
            break
        r_max = float(np.max(radial))
        if not np.isfinite(r_max) or r_max <= 0.0:
            continue
        dx = gx[remaining] - point[0]
        dy = gy[remaining] - point[1]
        in_box = (np.abs(dx) <= r_max) & (np.abs(dy) <= r_max)
        if not np.any(in_box):
            continue
        idx = remaining[in_box]
        dx = dx[in_box]
        dy = dy[in_box]
        dist2 = dx * dx + dy * dy
        in_circle = dist2 <= r_max * r_max
        if not np.any(in_circle):
            continue
        idx = idx[in_circle]
        dx = dx[in_circle]
        dy = dy[in_circle]
        dist2 = dist2[in_circle]
        bearing = np.mod(np.arctan2(dx, dy), _TWO_PI)
        q = bearing * angle_scale
        q_floor = np.floor(q)
        i0 = q_floor.astype(np.intp) % n_angle
        frac = q - q_floor
        i1 = (i0 + 1) % n_angle
        local_range = (1.0 - frac) * radial[i0] + frac * radial[i1]
        covered[idx[dist2 <= local_range * local_range]] = True
    return covered

def _polyline_min_distance_to_point(roads: np.ndarray, point: np.ndarray) -> float:
    a = roads[:-1]
    b = roads[1:]
    v = b - a
    vv = np.sum(v * v, axis=1)
    w = point[None, :] - a
    t = np.zeros(len(v), dtype=float)
    nonzero = vv > _EPS
    t[nonzero] = np.sum(w[nonzero] * v[nonzero], axis=1) / vv[nonzero]
    t = np.clip(t, 0.0, 1.0)
    closest = a + t[:, None] * v
    return float(np.min(np.linalg.norm(closest - point[None, :], axis=1)))

class PathObjectiveEvaluator:
    PENALTY_LENGTH = 1000000.0
    PENALTY_NO_NAV = 2000000.0
    PENALTY_SPACING = 3000000.0
    PENALTY_SHORT_SUBPATH = 4000000.0

    def __init__(self, sonar_matrix: np.ndarray, x_max: float, y_max: float, *, len_limit: float=12 * 1.852 * 24, n_d: int=100, d_min: float=2.0, sublen_min: float=5.0, no_nav: Optional[np.ndarray]=np.array([50.0, 50.0, 4.0]), coverage_nx: int=100, coverage_ny: int=100) -> None:
        self.sonar_matrix = np.asarray(sonar_matrix, dtype=float)
        if self.sonar_matrix.ndim != 3:
            raise ValueError('sonar_matrix 必须具有形状 (ny, nx, n_angle)。')
        self.x_max = float(x_max)
        self.y_max = float(y_max)
        self.len_limit = float(len_limit)
        self.n_d = int(n_d)
        self.d_min = float(d_min)
        self.sublen_min = float(sublen_min)
        self.no_nav = None if no_nav is None else np.asarray(no_nav, dtype=float).reshape(-1)
        if coverage_nx < 2 or coverage_ny < 2:
            raise ValueError('coverage_nx 和 coverage_ny 至少应为 2。')
        px = np.linspace(0.0, self.x_max, int(coverage_nx))
        py = np.linspace(0.0, self.y_max, int(coverage_ny))
        self.mpx, self.mpy = np.meshgrid(px, py, indexing='xy')
        self._grid_x_flat = self.mpx.ravel()
        self._grid_y_flat = self.mpy.ravel()

    def _geometry_and_constraint_penalty(self, heading_angle: float, path_positions: np.ndarray) -> tuple[PathGeometry, float, Optional[float]]:
        geometry = build_search_path(heading_angle, path_positions, self.x_max, self.y_max)
        if geometry.spacings.size and np.min(geometry.spacings) < self.d_min:
            return (geometry, np.nan, self.PENALTY_SPACING)
        if np.any(geometry.subpath_lengths < self.sublen_min):
            return (geometry, np.nan, self.PENALTY_SHORT_SUBPATH)
        _, _, total_length = sample_polyline(geometry.roads, 2)
        if total_length > self.len_limit:
            return (geometry, total_length, self.PENALTY_LENGTH)
        if self.no_nav is not None and self.no_nav.size >= 3 and (self.no_nav[2] > 0):
            distance = _polyline_min_distance_to_point(geometry.roads, self.no_nav[:2])
            if distance <= self.no_nav[2]:
                return (geometry, total_length, self.PENALTY_NO_NAV)
        return (geometry, total_length, None)

    def __call__(self, heading_angle: float, path_positions: np.ndarray) -> float:
        geometry, _, penalty = self._geometry_and_constraint_penalty(heading_angle, path_positions)
        if penalty is not None:
            return float(penalty)
        sample_points, _, _ = sample_polyline(geometry.roads, self.n_d)
        sample_ranges = bilinear_sonar_interpolation(sample_points, self.sonar_matrix, self.x_max, self.y_max)
        covered = calculate_coverage_mask(sample_points, sample_ranges, self._grid_x_flat, self._grid_y_flat)
        coverage_percent = 100.0 * float(np.mean(covered))
        return -coverage_percent

    def evaluate_details(self, heading_angle: float, path_positions: np.ndarray, *, enforce_constraints: bool=False) -> dict:
        geometry, total_length, penalty = self._geometry_and_constraint_penalty(heading_angle, path_positions)
        if enforce_constraints and penalty is not None:
            return {'geometry': geometry, 'roads': geometry.roads, 'total_length': total_length, 'penalty': penalty}
        sample_points, sample_headings, total_length = sample_polyline(geometry.roads, self.n_d)
        sample_ranges = bilinear_sonar_interpolation(sample_points, self.sonar_matrix, self.x_max, self.y_max)
        covered_flat = calculate_coverage_mask(sample_points, sample_ranges, self._grid_x_flat, self._grid_y_flat)
        coverage_mask = covered_flat.reshape(self.mpx.shape)
        return {'geometry': geometry, 'roads': geometry.roads, 'sample_points': sample_points, 'sample_headings': sample_headings, 'sample_ranges': sample_ranges, 'coverage_mask': coverage_mask, 'coverage_percent': 100.0 * float(np.mean(covered_flat)), 'total_length': total_length, 'penalty': penalty, 'mpx': self.mpx, 'mpy': self.mpy}

def objective_function(heading_angle, path_positions, sonar_matrix, X_MAX, Y_MAX, len_limit=12 * 1.852 * 24, n_d=100, d_min=2, sublen_min=5, no_nav=np.array([50, 50, 4]), *, coverage_nx=100, coverage_ny=100):
    evaluator = PathObjectiveEvaluator(sonar_matrix, X_MAX, Y_MAX, len_limit=len_limit, n_d=n_d, d_min=d_min, sublen_min=sublen_min, no_nav=no_nav, coverage_nx=coverage_nx, coverage_ny=coverage_ny)
    return evaluator(heading_angle, path_positions)
