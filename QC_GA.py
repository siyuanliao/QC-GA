"""本文使用的 QC-GA 优化算法。

算法按最小化问题实现，将种群划分为两个等规模子群，并由 Q-learning 动态选择遗传算子参数。
"""

import numpy as np

def _prepare_bounds(con_min, con_max, dim):
    lower = np.broadcast_to(np.asarray(con_min, dtype=float), (dim,)).copy()
    upper = np.broadcast_to(np.asarray(con_max, dtype=float), (dim,)).copy()
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
        raise ValueError('决策变量上下界必须为有限数值。')
    if np.any(upper <= lower):
        raise ValueError('每一维决策变量都必须满足 con_max > con_min。')
    return (lower, upper)

def _evaluate_population(population, obj_cal):
    return np.asarray([obj_cal(individual) for individual in population], dtype=float)

def _population_diversity(population):
    return float(np.mean(np.var(population, axis=0)))

def _get_diversity_state(diversity, low_threshold, high_threshold):
    if diversity >= high_threshold:
        return 0
    if diversity >= low_threshold:
        return 1
    return 2

def _choose_action(q_table, state, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(0, q_table.shape[1]))
    row = q_table[state]
    best_actions = np.flatnonzero(np.isclose(row, np.max(row)))
    return int(rng.choice(best_actions))

def _uniform_crossover(parent_a, parent_b, rng):
    mask = rng.random(parent_a.size) < 0.5
    return np.where(mask, parent_a, parent_b)

def _heuristic_crossover(parent_a, parent_b, fit_a, fit_b, rng):
    if fit_a <= fit_b:
        better, worse = (parent_a, parent_b)
    else:
        better, worse = (parent_b, parent_a)
    r = rng.random()
    return better + r * (better - worse)

def _mutate(child, mutation_rate, mutation_scale, span, rng):
    if rng.random() < mutation_rate:
        child = child + rng.normal(0.0, mutation_scale * span, size=child.size)
    return child

def _evolve_subpopulation(population, fitness, obj_cal, lower, upper, crossover_rate, mutation_rate, mutation_scale, crossover_mode, rng):
    pop_size, _ = population.shape
    span = upper - lower
    offspring = np.empty_like(population)
    for i in range(pop_size):
        mate_idx = int(rng.integers(0, pop_size - 1))
        if mate_idx >= i:
            mate_idx += 1
        target = population[i]
        mate = population[mate_idx]
        if rng.random() < crossover_rate:
            if crossover_mode == 'uniform':
                child = _uniform_crossover(target, mate, rng)
            elif crossover_mode == 'heuristic':
                child = _heuristic_crossover(target, mate, fitness[i], fitness[mate_idx], rng)
            else:
                raise ValueError(f'未知交叉方式: {crossover_mode}')
        else:
            child = target.copy()
        child = _mutate(child, mutation_rate, mutation_scale, span, rng)
        offspring[i] = np.clip(child, lower, upper)
    offspring_fitness = _evaluate_population(offspring, obj_cal)
    improved = offspring_fitness < fitness
    new_population = population.copy()
    new_fitness = fitness.copy()
    new_population[improved] = offspring[improved]
    new_fitness[improved] = offspring_fitness[improved]
    return (new_population, new_fitness)

def QC_GA(opt_vars, obj_cal, con_min, con_max, max_iter=100, *, epsilon=0.7, epsilon_min=0.05, epsilon_decay=0.97, learning_rate=0.1, discount_factor=0.9, migration_interval=10, diversity_thresholds=None, elite_ratio=0.1, migration_ratio=0.1, gaussian_sigma0=0.05, reward_max=10.0, reward_pos=1.0, reward_neg=-1.0, rng=None, verbose=True, return_diagnostics=False):
    opt_vars = np.asarray(opt_vars, dtype=float)
    if opt_vars.ndim != 2:
        raise ValueError('opt_vars 必须为二维数组，形状为 (pop_size, dim)。')
    pop_size, dim = opt_vars.shape
    if pop_size < 4:
        raise ValueError('种群规模至少应为 4。')
    if pop_size % 2 != 0:
        raise ValueError('两个协同子群采用等规模划分，因此 pop_size 必须为偶数。')
    if max_iter <= 0:
        raise ValueError('max_iter 必须为正整数。')
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError('epsilon 必须位于 [0, 1]。')
    if not 0.0 <= epsilon_min <= epsilon:
        raise ValueError('epsilon_min 必须满足 0 <= epsilon_min <= epsilon。')
    if not 0.0 < epsilon_decay <= 1.0:
        raise ValueError('epsilon_decay 必须满足 0 < epsilon_decay <= 1。')
    if not 0.0 < learning_rate < 1.0:
        raise ValueError('learning_rate 必须满足 0 < learning_rate < 1。')
    if not 0.0 <= discount_factor < 1.0:
        raise ValueError('discount_factor 必须满足 0 <= discount_factor < 1。')
    if migration_interval <= 0:
        raise ValueError('migration_interval 必须为正整数。')
    if not 0.0 < elite_ratio <= 1.0:
        raise ValueError('elite_ratio 必须位于 (0, 1]。')
    if not 0.0 < migration_ratio <= 1.0:
        raise ValueError('migration_ratio 必须位于 (0, 1]。')
    if gaussian_sigma0 < 0.0:
        raise ValueError('gaussian_sigma0 不能为负数。')
    if not reward_max > reward_pos > 0.0 > reward_neg:
        raise ValueError('奖励值必须满足 reward_max > reward_pos > 0 > reward_neg。')
    lower, upper = _prepare_bounds(con_min, con_max, dim)
    span = upper - lower
    if isinstance(rng, np.random.Generator):
        random_generator = rng
    else:
        random_generator = np.random.default_rng(rng)
    population = np.clip(opt_vars, lower, upper)
    half_size = pop_size // 2
    pop_explor = population[:half_size].copy()
    pop_exploit = population[half_size:].copy()
    fit_explor = _evaluate_population(pop_explor, obj_cal)
    fit_exploit = _evaluate_population(pop_exploit, obj_cal)
    explor_best_idx = int(np.argmin(fit_explor))
    exploit_best_idx = int(np.argmin(fit_exploit))
    if fit_explor[explor_best_idx] <= fit_exploit[exploit_best_idx]:
        global_best_obj = float(fit_explor[explor_best_idx])
        global_best_var = pop_explor[explor_best_idx].copy()
    else:
        global_best_obj = float(fit_exploit[exploit_best_idx])
        global_best_var = pop_exploit[exploit_best_idx].copy()
    previous_iter_best_obj = global_best_obj
    q_table = np.zeros((3, 4), dtype=float)
    initial_diversity = _population_diversity(population)
    if diversity_thresholds is None:
        low_threshold = 0.25 * initial_diversity
        high_threshold = 0.6 * initial_diversity
    else:
        thresholds = np.asarray(diversity_thresholds, dtype=float).reshape(-1)
        if thresholds.size != 2:
            raise ValueError('diversity_thresholds 必须为 (低阈值, 高阈值)。')
        low_threshold = float(thresholds[0])
        high_threshold = float(thresholds[1])
    if low_threshold < 0.0 or high_threshold <= low_threshold:
        raise ValueError('多样性阈值必须满足 0 <= 低阈值 < 高阈值。')
    num_local_elites = max(1, int(np.ceil(half_size * elite_ratio)))
    num_migrants = max(1, int(np.ceil(half_size * migration_ratio)))
    num_migrants = min(num_migrants, half_size)
    history_avg_obj = np.empty(max_iter, dtype=float)
    history_best_obj = np.empty(max_iter, dtype=float)
    state_history = np.empty(max_iter, dtype=np.intp)
    action_history = np.empty(max_iter, dtype=np.intp)
    reward_history = np.empty(max_iter, dtype=float)
    diversity_history = np.empty(max_iter, dtype=float)
    epsilon_history = np.empty(max_iter, dtype=float)
    q_table_history = np.empty((max_iter, 3, 4), dtype=float)
    action_parameters = {0: {'name': '强探索', 'pc_explor': 0.95, 'pm_explor': 0.3, 'pc_exploit': 0.75, 'pm_exploit': 0.15, 'mutation_scale': 0.15}, 1: {'name': '偏探索', 'pc_explor': 0.9, 'pm_explor': 0.2, 'pc_exploit': 0.7, 'pm_exploit': 0.1, 'mutation_scale': 0.1}, 2: {'name': '平衡搜索', 'pc_explor': 0.8, 'pm_explor': 0.15, 'pc_exploit': 0.85, 'pm_exploit': 0.08, 'mutation_scale': 0.05}, 3: {'name': '强开发', 'pc_explor': 0.7, 'pm_explor': 0.1, 'pc_exploit': 0.9, 'pm_exploit': 0.05, 'mutation_scale': 0.01}}
    for iteration in range(max_iter):
        k = iteration + 1
        if verbose and (iteration == 0 or k % 10 == 0 or k == max_iter):
            print(f'当前迭代进度：{k:d}/{max_iter:d}')
        current_pop = np.vstack((pop_explor, pop_exploit))
        diversity = _population_diversity(current_pop)
        state = _get_diversity_state(diversity, low_threshold, high_threshold)
        current_epsilon = max(epsilon_min, epsilon * epsilon_decay ** iteration)
        action = _choose_action(q_table, state, current_epsilon, random_generator)
        params = action_parameters[action]
        pop_explor, fit_explor = _evolve_subpopulation(pop_explor, fit_explor, obj_cal, lower, upper, params['pc_explor'], params['pm_explor'], params['mutation_scale'], 'uniform', random_generator)
        pop_exploit, fit_exploit = _evolve_subpopulation(pop_exploit, fit_exploit, obj_cal, lower, upper, params['pc_exploit'], params['pm_exploit'], params['mutation_scale'], 'heuristic', random_generator)
        if k % 5 == 0:
            elite_indices = np.argsort(fit_exploit)[:num_local_elites]
            sigma_ratio = gaussian_sigma0 * (1.0 - k / max_iter)
            if sigma_ratio > 0.0:
                sigma = sigma_ratio * span
                for idx in elite_indices:
                    neighbor = pop_exploit[idx] + random_generator.normal(0.0, sigma, size=dim)
                    neighbor = np.clip(neighbor, lower, upper)
                    neighbor_fit = float(obj_cal(neighbor))
                    if neighbor_fit < fit_exploit[idx]:
                        pop_exploit[idx] = neighbor
                        fit_exploit[idx] = neighbor_fit
        current_iter_best_obj = float(min(np.min(fit_explor), np.min(fit_exploit)))
        old_global_best_obj = global_best_obj
        if current_iter_best_obj < old_global_best_obj:
            reward = reward_max
        elif old_global_best_obj <= current_iter_best_obj < previous_iter_best_obj:
            reward = reward_pos
        else:
            reward = reward_neg
        next_population = np.vstack((pop_explor, pop_exploit))
        next_diversity = _population_diversity(next_population)
        next_state = _get_diversity_state(next_diversity, low_threshold, high_threshold)
        q_table[state, action] += learning_rate * (reward + discount_factor * np.max(q_table[next_state]) - q_table[state, action])
        if current_iter_best_obj < global_best_obj:
            if np.min(fit_explor) <= np.min(fit_exploit):
                best_idx = int(np.argmin(fit_explor))
                global_best_var = pop_explor[best_idx].copy()
                global_best_obj = float(fit_explor[best_idx])
            else:
                best_idx = int(np.argmin(fit_exploit))
                global_best_var = pop_exploit[best_idx].copy()
                global_best_obj = float(fit_exploit[best_idx])
        if k % migration_interval == 0:
            explor_best_indices = np.argsort(fit_explor)[:num_migrants]
            exploit_worst_indices = np.argsort(fit_exploit)[-num_migrants:]
            pop_exploit[exploit_worst_indices] = pop_explor[explor_best_indices].copy()
            fit_exploit[exploit_worst_indices] = fit_explor[explor_best_indices].copy()
        current_fitness = np.concatenate((fit_explor, fit_exploit))
        history_avg_obj[iteration] = float(np.mean(current_fitness))
        history_best_obj[iteration] = global_best_obj
        state_history[iteration] = state
        action_history[iteration] = action
        reward_history[iteration] = reward
        diversity_history[iteration] = diversity
        epsilon_history[iteration] = current_epsilon
        q_table_history[iteration] = q_table
        previous_iter_best_obj = current_iter_best_obj
    current_vars = np.vstack((pop_explor, pop_exploit))
    result = (global_best_var, global_best_obj, current_vars, history_avg_obj, history_best_obj)
    if not return_diagnostics:
        return result
    diagnostics = {'q_table': q_table.copy(), 'q_table_history': q_table_history, 'state_history': state_history, 'action_history': action_history, 'reward_history': reward_history, 'diversity_history': diversity_history, 'epsilon_history': epsilon_history, 'diversity_thresholds': np.array([low_threshold, high_threshold], dtype=float), 'initial_diversity': initial_diversity, 'action_parameters': action_parameters, 'state_names': ('高多样性', '中等多样性', '低多样性'), 'action_names': ('强探索', '偏探索', '平衡搜索', '强开发')}
    return result + (diagnostics,)
