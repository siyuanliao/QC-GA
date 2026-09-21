"""I-LNS 对比算法。

这里使用适配连续变量问题的破坏—修复形式，用于本文路径规划对比实验。
"""

import numpy as np

def roulette(weights):
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.sum(weights)
    return np.random.choice(len(weights), p=weights)

def random_destroy(x, rate=0.2):
    y = x.copy()
    mask = np.random.rand(len(x)) < rate
    if not np.any(mask):
        mask[np.random.randint(len(x))] = True
    pool = np.where(mask)[0]
    return (y, pool)

def worst_destroy(x, obj_cal, rate=0.2):
    y = x.copy()
    n_remove = max(1, int(len(x) * rate))
    scores = []
    base = obj_cal(x)
    for i in range(len(x)):
        tmp = x.copy()
        tmp[i] = np.random.uniform(np.min(x), np.max(x))
        scores.append(obj_cal(tmp) - base)
    pool = np.argsort(scores)[-n_remove:]
    return (y, pool)

def repair(x, pool, lb, ub, obj_cal):
    best = x.copy()
    for index in pool:
        candidates = []
        for _ in range(10):
            temp = best.copy()
            temp[index] = np.random.uniform(lb[index], ub[index])
            candidates.append(temp)
        values = [obj_cal(c) for c in candidates]
        best = candidates[np.argmin(values)]
    return best

def adaptive_weight(weights, index, flag):
    if flag:
        weights[index] /= 0.8
    else:
        weights[index] *= 0.8
    weights = np.maximum(weights, 1e-08)
    return weights

def I_LNS(opt_vars, obj_cal, con_min, con_max, max_iter=100):
    X = np.asarray(opt_vars, dtype=float).copy()
    lb = np.asarray(con_min)
    ub = np.asarray(con_max)
    if lb.ndim == 0:
        lb = np.ones(X.shape[1]) * lb
    if ub.ndim == 0:
        ub = np.ones(X.shape[1]) * ub
    X = np.clip(X, lb, ub)
    current = X[np.argmin([obj_cal(x) for x in X])].copy()
    best = current.copy()
    best_obj = obj_cal(best)
    S_W = np.ones(3)
    P_W = np.ones(2)
    history_avg = []
    history_best = []
    archive = []
    for it in range(max_iter):
        structure = roulette(S_W)
        pattern = roulette(P_W)
        flag = 0
        if pattern == 0:
            candidate, pool = random_destroy(current, 0.2)
        else:
            candidate, pool = worst_destroy(current, obj_cal, 0.2)
        candidate = repair(candidate, pool, lb, ub, obj_cal)
        candidate = np.clip(candidate, lb, ub)
        f_old = obj_cal(current)
        f_new = obj_cal(candidate)
        if f_new < f_old:
            current = candidate.copy()
            flag = 1
        if f_new < best_obj:
            best = candidate.copy()
            best_obj = f_new
        S_W = adaptive_weight(S_W, structure, flag)
        P_W = adaptive_weight(P_W, pattern, flag)
        archive.append(best.copy())
        history_avg.append(np.mean([obj_cal(a) for a in archive]))
        history_best.append(best_obj)
    return (best, best_obj, np.asarray(archive), history_avg, history_best)
