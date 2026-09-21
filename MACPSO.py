"""MACPSO 对比算法。

实现采用多 Actor、单 Critic 的粒子群结构，与论文对比实验中的设置一致。
"""

from collections import deque
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def _prepare_bounds(con_min, con_max, dim):
    lower = np.broadcast_to(np.asarray(con_min, dtype=float), (dim,)).copy()
    upper = np.broadcast_to(np.asarray(con_max, dtype=float), (dim,)).copy()
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
        raise ValueError('决策变量上下界必须为有限数值。')
    if np.any(upper <= lower):
        raise ValueError('每一维决策变量必须满足 con_max > con_min。')
    return (lower, upper)

def _safe_objective(obj_cal, x):
    value = float(obj_cal(x))
    return value if np.isfinite(value) else np.inf

def _evaluate_population(population, obj_cal):
    return np.asarray([_safe_objective(obj_cal, individual) for individual in population], dtype=float)

def _truncated_cauchy(loc, scale, rng, max_trials=100):
    loc = np.asarray(loc, dtype=float)
    result = np.empty_like(loc)
    flat_loc = loc.reshape(-1)
    flat_result = result.reshape(-1)
    for i, mu in enumerate(flat_loc):
        value = np.nan
        for _ in range(max_trials):
            candidate = mu + scale * rng.standard_cauchy()
            if 0.0 <= candidate <= 1.0:
                value = candidate
                break
        if not np.isfinite(value):
            value = np.clip(mu, 0.0, 1.0)
        flat_result[i] = value
    return result

def _soft_update(target_net, source_net, tau):
    with torch.no_grad():
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(tau * source_param.data)

class _Actor(nn.Module):

    def __init__(self, state_dim=3, action_dim=4, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.LeakyReLU(), nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(), nn.Linear(hidden_dim, action_dim), nn.Sigmoid())

    def forward(self, state):
        return self.net(state)

class _Critic(nn.Module):

    def __init__(self, state_dim=3, action_dim=4, hidden_dim=256):
        super().__init__()
        self.feature = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim), nn.LeakyReLU(), nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.q_head = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=-1)
        return self.q_head(self.feature(x))

class _ReplayBuffer:

    def __init__(self, capacity):
        self.buffer = deque(maxlen=int(capacity))

    def __len__(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state, group_id):
        self.buffer.append((np.asarray(state, dtype=np.float32).copy(), np.asarray(action, dtype=np.float32).copy(), float(reward), np.asarray(next_state, dtype=np.float32).copy(), int(group_id)))

    def sample(self, batch_size, rng):
        indices = rng.choice(len(self.buffer), size=batch_size, replace=False)
        states = []
        actions = []
        rewards = []
        next_states = []
        group_ids = []
        for idx in indices:
            s, a, r, ns, gid = self.buffer[int(idx)]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            group_ids.append(gid)
        return (np.asarray(states, dtype=np.float32), np.asarray(actions, dtype=np.float32), np.asarray(rewards, dtype=np.float32), np.asarray(next_states, dtype=np.float32), np.asarray(group_ids, dtype=np.int64))

def _build_states(population, fitness, groups, historical_worst_pos, historical_worst_fit, global_best_pos, global_best_fit, iteration, max_iter):
    n_particles = population.shape[0]
    states = np.zeros((n_particles, 3), dtype=np.float32)
    delta_p_global = float(np.linalg.norm(historical_worst_pos - global_best_pos))
    delta_f_global = float(historical_worst_fit - global_best_fit)
    eps = 1e-12
    delta_p_global = max(delta_p_global, eps)
    delta_f_global = max(delta_f_global, eps)
    for group_indices in groups:
        group_fit = fitness[group_indices]
        local_best_local_idx = int(np.argmin(group_fit))
        local_best_idx = int(group_indices[local_best_local_idx])
        group_best_pos = population[local_best_idx]
        group_best_fit = float(fitness[local_best_idx])
        for idx in group_indices:
            delta_p = float(np.linalg.norm(population[idx] - group_best_pos))
            delta_f = float(fitness[idx] - group_best_fit)
            states[idx, 0] = delta_p / delta_p_global
            states[idx, 1] = delta_f / delta_f_global
            states[idx, 2] = float(iteration) / float(max_iter)
    states = np.nan_to_num(states, nan=0.0, posinf=1000000.0, neginf=-1000000.0)
    return states

def _train_ddpg(replay_buffer, actors, target_actors, critic, target_critic, actor_optimizers, critic_optimizer, batch_size, gamma, tau, device, rng):
    if len(replay_buffer) < batch_size:
        return (None, [None] * len(actors))
    states, actions, rewards, next_states, group_ids = replay_buffer.sample(batch_size, rng)
    states_t = torch.as_tensor(states, dtype=torch.float32, device=device)
    actions_t = torch.as_tensor(actions, dtype=torch.float32, device=device)
    rewards_t = torch.as_tensor(rewards.reshape(-1, 1), dtype=torch.float32, device=device)
    next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=device)
    group_ids_t = torch.as_tensor(group_ids, dtype=torch.long, device=device)
    with torch.no_grad():
        next_actions_t = torch.zeros((batch_size, 4), dtype=torch.float32, device=device)
        for group_id, target_actor in enumerate(target_actors):
            mask = group_ids_t == group_id
            if torch.any(mask):
                next_actions_t[mask] = target_actor(next_states_t[mask])
        target_q = rewards_t + gamma * target_critic(next_states_t, next_actions_t)
    current_q = critic(states_t, actions_t)
    critic_loss = torch.mean((current_q - target_q) ** 2)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=10.0)
    critic_optimizer.step()
    actor_losses = []
    for param in critic.parameters():
        param.requires_grad_(False)
    for group_id, actor in enumerate(actors):
        mask = group_ids_t == group_id
        if not torch.any(mask):
            actor_losses.append(None)
            continue
        group_states = states_t[mask]
        predicted_actions = actor(group_states)
        actor_loss = -critic(group_states, predicted_actions).mean()
        actor_optimizers[group_id].zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=10.0)
        actor_optimizers[group_id].step()
        actor_losses.append(float(actor_loss.detach().cpu().item()))
    for param in critic.parameters():
        param.requires_grad_(True)
    _soft_update(target_critic, critic, tau)
    for target_actor, actor in zip(target_actors, actors):
        _soft_update(target_actor, actor, tau)
    return (float(critic_loss.detach().cpu().item()), actor_losses)

def MACPSO(opt_vars, obj_cal, con_min, con_max, max_iter=100, *, n_groups=3, k=3, omega0=1.0, alpha=0.99, cauchy_sigma=0.2, actor_lr=0.001, critic_lr=0.001, replay_capacity=50000, batch_size=64, gamma=0.99, tau=0.005, hidden_dim=256, rng=None, verbose=True, return_diagnostics=False, device=None):
    population = np.asarray(opt_vars, dtype=float).copy()
    if population.ndim != 2:
        raise ValueError('opt_vars 必须是二维数组，形状为 (pop_size, dim)。')
    pop_size, dim = population.shape
    if pop_size < n_groups * 2:
        raise ValueError('粒子数过少，每个粒子组至少需要2个粒子。')
    if not 1 <= n_groups <= pop_size // 2:
        raise ValueError('n_groups 设置不合理。')
    if k <= 0:
        raise ValueError('k 必须为正整数。')
    if max_iter <= 0:
        raise ValueError('max_iter 必须为正整数。')
    if batch_size <= 0:
        raise ValueError('batch_size 必须为正整数。')
    if replay_capacity < batch_size:
        raise ValueError('replay_capacity 必须不小于 batch_size。')
    if not 0.0 <= gamma < 1.0:
        raise ValueError('gamma 必须位于 [0,1)。')
    if not 0.0 < tau <= 1.0:
        raise ValueError('tau 必须位于 (0,1]。')
    lower, upper = _prepare_bounds(con_min, con_max, dim)
    if isinstance(rng, np.random.Generator):
        random_generator = rng
        torch_seed = int(random_generator.integers(0, 2 ** 31 - 1))
    else:
        random_generator = np.random.default_rng(rng)
        torch_seed = 0 if rng is None else int(rng)
    torch.manual_seed(torch_seed)
    if device is None:
        torch_device = torch.device('cpu')
    else:
        torch_device = torch.device(device)
    population = np.clip(population, lower, upper)
    groups = [np.asarray(indices, dtype=np.int64) for indices in np.array_split(np.arange(pop_size), n_groups)]
    group_of_particle = np.empty(pop_size, dtype=np.int64)
    for group_id, group_indices in enumerate(groups):
        group_of_particle[group_indices] = group_id
    velocity = random_generator.uniform(-1.0, 1.0, size=(pop_size, dim))
    fitness = _evaluate_population(population, obj_cal)
    function_evaluations = pop_size
    personal_best_pos = population.copy()
    personal_best_fit = fitness.copy()
    global_best_idx = int(np.argmin(personal_best_fit))
    global_best_pos = personal_best_pos[global_best_idx].copy()
    global_best_fit = float(personal_best_fit[global_best_idx])
    historical_worst_idx = int(np.argmax(fitness))
    historical_worst_pos = population[historical_worst_idx].copy()
    historical_worst_fit = float(fitness[historical_worst_idx])
    actors = [_Actor(hidden_dim=hidden_dim).to(torch_device) for _ in range(n_groups)]
    target_actors = [copy.deepcopy(actor).to(torch_device) for actor in actors]
    critic = _Critic(hidden_dim=hidden_dim).to(torch_device)
    target_critic = copy.deepcopy(critic).to(torch_device)
    actor_optimizers = [optim.Adam(actor.parameters(), lr=actor_lr) for actor in actors]
    critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)
    replay_buffer = _ReplayBuffer(replay_capacity)
    history_avg_obj = np.empty(max_iter, dtype=float)
    history_best_obj = np.empty(max_iter, dtype=float)
    critic_loss_history = np.full(max_iter, np.nan, dtype=float)
    actor_loss_history = np.full((max_iter, n_groups), np.nan, dtype=float)
    omega_history = np.empty(max_iter, dtype=float)
    mean_action_history = np.empty((max_iter, 4), dtype=float)
    for iteration in range(1, max_iter + 1):
        if verbose and (iteration == 1 or iteration % 10 == 0 or iteration == max_iter):
            print(f'MACPSO 当前迭代进度：{iteration}/{max_iter}')
        omega_t = omega0 * alpha ** iteration
        omega_history[iteration - 1] = omega_t
        previous_personal_best_fit = personal_best_fit.copy()
        states = _build_states(population, fitness, groups, historical_worst_pos, historical_worst_fit, global_best_pos, global_best_fit, iteration - 1, max_iter)
        actions = np.zeros((pop_size, 4), dtype=float)
        with torch.no_grad():
            for group_id, group_indices in enumerate(groups):
                state_tensor = torch.as_tensor(states[group_indices], dtype=torch.float32, device=torch_device)
                action_tensor = actors[group_id](state_tensor)
                actions[group_indices] = action_tensor.cpu().numpy()
        mean_action_history[iteration - 1] = np.mean(actions, axis=0)
        rc = _truncated_cauchy(actions, cauchy_sigma, random_generator)
        c1 = 1.0 - omega_t * rc[:, 0]
        c2 = (1.0 - omega_t) * rc[:, 1]
        c3 = omega_t * rc[:, 2]
        c4 = (1.0 - omega_t) * rc[:, 3]
        new_velocity = velocity.copy()
        for group_indices in groups:
            group_best_local_idx = int(np.argmin(personal_best_fit[group_indices]))
            group_best_idx = int(group_indices[group_best_local_idx])
            p_group = personal_best_pos[group_best_idx]
            for idx in group_indices:
                selectable = group_indices[group_indices != idx]
                if selectable.size >= 2:
                    r1_idx, r2_idx = random_generator.choice(selectable, size=2, replace=False)
                else:
                    r1_idx, r2_idx = random_generator.choice(group_indices, size=2, replace=True)
                new_velocity[idx] = omega_t * velocity[idx] + c1[idx] * (personal_best_pos[idx] - population[idx]) + c2[idx] * (p_group - population[idx]) + c2[idx] * (personal_best_pos[int(r1_idx)] - personal_best_pos[int(r2_idx)])
        velocity = new_velocity
        population = np.clip(population + velocity, lower, upper)
        fitness = _evaluate_population(population, obj_cal)
        function_evaluations += pop_size
        improved = fitness < personal_best_fit
        personal_best_pos[improved] = population[improved]
        personal_best_fit[improved] = fitness[improved]
        global_best_idx = int(np.argmin(personal_best_fit))
        global_best_pos = personal_best_pos[global_best_idx].copy()
        global_best_fit = float(personal_best_fit[global_best_idx])
        if iteration % k == 0:
            inter_velocity = velocity.copy()
            for idx in range(pop_size):
                selectable = np.arange(pop_size)
                selectable = selectable[selectable != idx]
                r3_idx, r4_idx = random_generator.choice(selectable, size=2, replace=False)
                inter_velocity[idx] = omega_t * velocity[idx] + c3[idx] * (personal_best_pos[idx] - population[idx]) + c4[idx] * (global_best_pos - population[idx]) + c4[idx] * (personal_best_pos[int(r3_idx)] - personal_best_pos[int(r4_idx)])
            velocity = inter_velocity
            population = np.clip(population + velocity, lower, upper)
            fitness = _evaluate_population(population, obj_cal)
            function_evaluations += pop_size
            improved = fitness < personal_best_fit
            personal_best_pos[improved] = population[improved]
            personal_best_fit[improved] = fitness[improved]
            global_best_idx = int(np.argmin(personal_best_fit))
            global_best_pos = personal_best_pos[global_best_idx].copy()
            global_best_fit = float(personal_best_fit[global_best_idx])
            for group_indices in groups:
                worst_local_idx = int(np.argmax(fitness[group_indices]))
                worst_idx = int(group_indices[worst_local_idx])
                velocity[worst_idx] = random_generator.uniform(-1.0, 1.0, size=dim)
                population[worst_idx] = random_generator.uniform(lower, upper, size=dim)
                fitness[worst_idx] = _safe_objective(obj_cal, population[worst_idx])
                function_evaluations += 1
                if fitness[worst_idx] < personal_best_fit[worst_idx]:
                    personal_best_fit[worst_idx] = fitness[worst_idx]
                    personal_best_pos[worst_idx] = population[worst_idx].copy()
            global_best_idx = int(np.argmin(personal_best_fit))
            global_best_pos = personal_best_pos[global_best_idx].copy()
            global_best_fit = float(personal_best_fit[global_best_idx])
        current_worst_idx = int(np.argmax(fitness))
        current_worst_fit = float(fitness[current_worst_idx])
        if current_worst_fit > historical_worst_fit:
            historical_worst_fit = current_worst_fit
            historical_worst_pos = population[current_worst_idx].copy()
        rewards = previous_personal_best_fit - fitness
        next_states = _build_states(population, fitness, groups, historical_worst_pos, historical_worst_fit, global_best_pos, global_best_fit, iteration, max_iter)
        for idx in range(pop_size):
            replay_buffer.add(states[idx], actions[idx], rewards[idx], next_states[idx], group_of_particle[idx])
        critic_loss, actor_losses = _train_ddpg(replay_buffer, actors, target_actors, critic, target_critic, actor_optimizers, critic_optimizer, min(batch_size, len(replay_buffer)), gamma, tau, torch_device, random_generator)
        if critic_loss is not None:
            critic_loss_history[iteration - 1] = critic_loss
        for group_id, loss_value in enumerate(actor_losses):
            if loss_value is not None:
                actor_loss_history[iteration - 1, group_id] = loss_value
        history_avg_obj[iteration - 1] = float(np.mean(fitness))
        history_best_obj[iteration - 1] = global_best_fit
    current_vars = population.copy()
    result = (global_best_pos.copy(), global_best_fit, current_vars, history_avg_obj, history_best_obj)
    if not return_diagnostics:
        return result
    diagnostics = {'function_evaluations': int(function_evaluations), 'n_groups': int(n_groups), 'k': int(k), 'omega_history': omega_history, 'mean_action_history': mean_action_history, 'critic_loss_history': critic_loss_history, 'actor_loss_history': actor_loss_history, 'replay_size': len(replay_buffer), 'paper_parameters': {'n_groups': n_groups, 'k': k, 'omega0': omega0, 'alpha': alpha, 'cauchy_sigma': cauchy_sigma, 'actor_lr': actor_lr, 'critic_lr': critic_lr, 'replay_capacity': replay_capacity, 'hidden_dim': hidden_dim}, 'implementation_parameters_not_explicit_in_paper': {'batch_size': batch_size, 'gamma': gamma, 'tau': tau}}
    return result + (diagnostics,)
