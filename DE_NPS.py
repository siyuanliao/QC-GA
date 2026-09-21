"""DE-NPS 对比算法。

保留 current-to-pbest/1 变异、F/CR 自适应记忆、螺旋扰动和停滞个体增强机制。
"""

import numpy as np

def DE_NPS(opt_vars, obj_cal, con_min, con_max, max_iter=100, H=4, p_best=0.2, tau=0.1, stagnation_limit=None):
    X = np.asarray(opt_vars, dtype=float).copy()
    NP, D = X.shape
    lb = np.asarray(con_min, dtype=float)
    ub = np.asarray(con_max, dtype=float)
    if lb.ndim == 0:
        lb = np.ones(D) * lb
    if ub.ndim == 0:
        ub = np.ones(D) * ub
    X = np.clip(X, lb, ub)

    def evaluate(pop):
        return np.array([obj_cal(x) for x in pop])
    Fval = evaluate(X)
    best_id = np.argmin(Fval)
    best_x = X[best_id].copy()
    best_f = Fval[best_id]
    memory_F = np.ones(H) * 0.5
    memory_CR = np.ones(H) * 0.5
    history_avg = [np.mean(Fval)]
    history_best = [best_f]
    if stagnation_limit is None:
        stagnation_limit = 2 * D
    stagnation = np.zeros(NP)
    for g in range(max_iter):
        new_X = np.zeros_like(X)
        used_F = np.zeros(NP)
        used_CR = np.zeros(NP)
        for i in range(NP):
            index = np.random.randint(H)
            if g < 0.2 * max_iter:
                F = memory_F[index] + 0.1 * np.random.rand()
            else:
                F = np.random.standard_cauchy() * 0.1 + memory_F[index]
            F = np.clip(F, 0, 1)
            CR = np.random.normal(memory_CR[index], 0.1)
            CR = np.clip(CR, 0, 1)
            used_F[i] = F
            used_CR[i] = CR
            order = np.argsort(Fval)
            p_num = max(2, int(NP * p_best))
            pbest = X[np.random.choice(order[:p_num])]
            candidates = list(range(NP))
            candidates.remove(i)
            r1, r2 = np.random.choice(candidates, 2, replace=False)
            mutant = X[i] + F * (pbest - X[i]) + F * (X[r1] - X[r2])
            mutant = np.clip(mutant, lb, ub)
            trial = X[i].copy()
            for j in range(D):
                if np.random.rand() < CR or j == np.random.randint(D):
                    trial[j] = mutant[j]
                elif np.random.rand() < tau:
                    a = -1 + g / max_iter
                    l = (a - 1) * np.random.rand() + 1
                    b = 1
                    dis = mutant[j] - X[i, j]
                    trial[j] = dis * np.exp(l * b) * np.cos(l * 2 * np.pi) + X[i, j]
            new_X[i] = np.clip(trial, lb, ub)
        new_F = evaluate(new_X)
        success_F = []
        success_CR = []
        improvements = []
        for i in range(NP):
            if new_F[i] < Fval[i]:
                success_F.append(used_F[i])
                success_CR.append(used_CR[i])
                improvements.append(Fval[i] - new_F[i])
                X[i] = new_X[i]
                Fval[i] = new_F[i]
                stagnation[i] = 0
            else:
                stagnation[i] += 1
        if len(success_F) > 0:
            w = np.asarray(improvements)
            w = w / np.sum(w)
            memory_F[np.random.randint(H)] = np.sum(w * np.asarray(success_F) ** 2) / np.sum(w * np.asarray(success_F))
            memory_CR[np.random.randint(H)] = np.sum(w * np.asarray(success_CR) ** 2) / np.sum(w * np.asarray(success_CR))
        for i in range(NP):
            if stagnation[i] > stagnation_limit:
                r1, r2 = np.random.choice(NP, 2, replace=False)
                if np.random.rand() < 0.5:
                    X[i] = X[i] + np.random.rand() * (X[r1] - best_x)
                else:
                    X[i] = best_x + np.random.rand() * (X[r1] - X[r2])
                X[i] = np.clip(X[i], lb, ub)
                Fval[i] = obj_cal(X[i])
                stagnation[i] = 0
        idx = np.argmin(Fval)
        if Fval[idx] < best_f:
            best_f = Fval[idx]
            best_x = X[idx].copy()
        history_avg.append(np.mean(Fval))
        history_best.append(best_f)
    return (best_x, best_f, X, history_avg, history_best)
