"""KPMBO 对比算法。

采用 KPCA、MIMIC 风格采样和 GP-UCB，并使用批量真实评价以便与种群算法保持接近的评价预算。
"""

import numpy as np

def _prepare_bounds(con_min, con_max, dim):
    lower = np.broadcast_to(np.asarray(con_min, dtype=float), (dim,)).copy()
    upper = np.broadcast_to(np.asarray(con_max, dtype=float), (dim,)).copy()
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
        raise ValueError('决策变量上下界必须为有限数值。')
    if np.any(upper <= lower):
        raise ValueError('每一维都必须满足 con_max > con_min。')
    return (lower, upper)

def rbf_kernel(X1, X2, length=1.0):
    X1 = np.atleast_2d(np.asarray(X1, dtype=float))
    X2 = np.atleast_2d(np.asarray(X2, dtype=float))
    dist2 = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2)
    length = max(float(length), 1e-12)
    return np.exp(-dist2 / (2.0 * length ** 2))

def kpca_projection(X, dim):
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    dim = int(max(1, min(dim, n - 1 if n > 1 else 1)))
    K = rbf_kernel(X, X)
    one = np.ones((n, n), dtype=float) / n
    Kc = K - one @ K - K @ one + one @ K @ one
    eigvals, eigvecs = np.linalg.eigh(Kc)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    eigvals = np.maximum(eigvals[:dim], 1e-12)
    eigvecs = eigvecs[:, :dim]
    return eigvecs * np.sqrt(eigvals)

def gp_predict(X_train, y_train, X_test, noise=1e-06):
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    X_test = np.asarray(X_test, dtype=float)
    K = rbf_kernel(X_train, X_train)
    K = K + float(noise) * np.eye(K.shape[0])
    Ks = rbf_kernel(X_train, X_test)
    try:
        alpha = np.linalg.solve(K, y_train)
        solve_Ks = np.linalg.solve(K, Ks)
    except np.linalg.LinAlgError:
        K = K + 1e-06 * np.eye(K.shape[0])
        alpha = np.linalg.solve(K, y_train)
        solve_Ks = np.linalg.solve(K, Ks)
    mu = Ks.T @ alpha
    var = 1.0 - np.sum(Ks * solve_Ks, axis=0)
    sigma = np.sqrt(np.maximum(var, 1e-12))
    return (mu, sigma)

def mimic_sample(Z, num):
    Z = np.asarray(Z, dtype=float)
    num = int(num)
    if Z.ndim != 2 or Z.shape[0] < 2:
        return np.random.normal(0.0, 1.0, size=(num, Z.shape[1]))
    mu = np.mean(Z, axis=0)
    if Z.shape[1] == 1:
        variance = float(np.var(Z[:, 0])) + 1e-06
        return np.random.normal(mu[0], np.sqrt(variance), size=(num, 1))
    cov = np.cov(Z.T)
    cov = np.atleast_2d(cov)
    cov += 1e-06 * np.eye(Z.shape[1])
    try:
        return np.random.multivariate_normal(mu, cov, num)
    except np.linalg.LinAlgError:
        std = np.sqrt(np.maximum(np.diag(cov), 1e-06))
        return mu + np.random.normal(0.0, std, size=(num, Z.shape[1]))

def _build_candidate_pool(X, Z, elite_Z, lower, upper, candidate_count, iteration, max_iter):
    dim = X.shape[1]
    candidate_Z = mimic_sample(elite_Z, candidate_count)
    candidates = np.empty((candidate_count, dim), dtype=float)
    progress = iteration / max(max_iter - 1, 1)
    perturb_ratio = 0.15 * (1.0 - progress) + 0.02
    span = upper - lower
    for idx, z in enumerate(candidate_Z):
        nearest = int(np.argmin(np.linalg.norm(Z - z, axis=1)))
        x = X[nearest].copy()
        z_distance = float(np.linalg.norm(Z[nearest] - z))
        scale = perturb_ratio * (1.0 + min(z_distance, 2.0))
        x += np.random.normal(0.0, scale, size=dim) * span
        candidates[idx] = np.clip(x, lower, upper)
    return candidates

def _select_active_set(X_all, y_all, active_size):
    active_size = min(int(active_size), len(X_all))
    order = np.argsort(y_all)[::-1]
    elite_num = max(1, int(np.ceil(0.7 * active_size)))
    elite_idx = order[:elite_num]
    if elite_num >= active_size:
        keep = elite_idx[:active_size]
        return (X_all[keep].copy(), y_all[keep].copy())
    remainder = order[elite_num:]
    need = active_size - elite_num
    if len(remainder) <= need:
        diversity_idx = remainder
    else:
        diversity_idx = np.random.choice(remainder, size=need, replace=False)
    keep = np.concatenate((elite_idx, np.asarray(diversity_idx, dtype=int)))
    return (X_all[keep].copy(), y_all[keep].copy())

def KPMBO(opt_vars, obj_cal, con_min, con_max, max_iter=100, *, batch_size=None, candidate_multiplier=3, reduced_dim=None, beta_scale=2.0):
    X = np.asarray(opt_vars, dtype=float).copy()
    if X.ndim != 2:
        raise ValueError('opt_vars 必须为二维数组，形状为 (pop_size, dim)。')
    pop_size, dim = X.shape
    if pop_size < 2:
        raise ValueError('KPMBO 至少需要 2 个初始样本。')
    if max_iter <= 0:
        raise ValueError('max_iter 必须为正整数。')
    lower, upper = _prepare_bounds(con_min, con_max, dim)
    X = np.clip(X, lower, upper)
    if batch_size is None:
        batch_size = pop_size
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError('batch_size 必须为正整数。')
    candidate_multiplier = max(1, int(candidate_multiplier))
    candidate_count = max(batch_size, candidate_multiplier * batch_size)
    if reduced_dim is None:
        reduced_dim = max(2, min(5, dim // 2))
    reduced_dim = max(1, min(int(reduced_dim), dim, pop_size - 1))
    y = -np.asarray([obj_cal(x) for x in X], dtype=float)
    best_index = int(np.argmax(y))
    best_x = X[best_index].copy()
    best_y = float(y[best_index])
    history_avg_obj = np.empty(max_iter, dtype=float)
    history_best_obj = np.empty(max_iter, dtype=float)
    for iteration in range(max_iter):
        Z = kpca_projection(X, reduced_dim)
        elite_num = max(5, pop_size // 5)
        elite_num = min(elite_num, pop_size)
        elite_idx = np.argsort(y)[-elite_num:]
        elite_Z = Z[elite_idx]
        candidates = _build_candidate_pool(X, Z, elite_Z, lower, upper, candidate_count, iteration, max_iter)
        mu, sigma = gp_predict(X, y, candidates)
        beta = float(beta_scale) * np.log(iteration + 2.0)
        ucb = mu + np.sqrt(max(beta, 0.0)) * sigma
        selected_idx = np.argsort(ucb)[-batch_size:]
        selected_candidates = candidates[selected_idx]
        selected_obj = np.asarray([obj_cal(x) for x in selected_candidates], dtype=float)
        selected_y = -selected_obj
        local_best_idx = int(np.argmax(selected_y))
        if selected_y[local_best_idx] > best_y:
            best_y = float(selected_y[local_best_idx])
            best_x = selected_candidates[local_best_idx].copy()
        X_all = np.vstack((X, selected_candidates))
        y_all = np.concatenate((y, selected_y))
        X, y = _select_active_set(X_all, y_all, pop_size)
        history_avg_obj[iteration] = float(-np.mean(y))
        history_best_obj[iteration] = float(-best_y)
    return (best_x, float(-best_y), X, history_avg_obj, history_best_obj)
