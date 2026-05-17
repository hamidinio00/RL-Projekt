# hierarchical_learning/hrl_training.py
import torch
import torch.optim as optim
import numpy as np
import csv
import time
import os
import multiprocessing as mp

# Eigene Module
from simulation.internal import Plan
from hierarchical_learning.hierarchical_vec_env import SubprocVecEnv
from hierarchical_learning.hierarichical_agent import ManagerNetwork, WorkerNetwork
from learning.utils import RunningMeanStd

# --- HYPERPARAMETER ---
HRL_CONFIG = {
    'n_envs': 6,
    'total_timesteps': 350,
    'total_timesteps_math': 500,
    'steps_per_update': 256,

    # --- EXPERIMENT MODUS ---
    'use_implicit': True,  # True = Prioritäten (Implizit), False = Strategie (Explizit)

    # --- SCHEDULING CONFIG ---
    'decay_start_worker': 50,
    'decay_start_manager': 50,

    # Learning Rates
    'lr_worker_start': 3e-4,
    'lr_worker_end': 3e-5,

    'lr_manager_start': 3e-4,
    'lr_manager_end': 3e-5,

    # Entropy (Exploration)
    'ent_worker_start': 0.03,
    'ent_worker_end': 0.001,

    'ent_manager_start': 0.03,
    'ent_manager_end': 0.001,

    # PPO Params
    'gamma_worker': 0.99,
    'gamma_manager': 0.995,
    'clip_epsilon': 0.2,
    'gae_lambda': 0.95,
    'max_grad_norm': 0.5,
    'n_epochs': 4,
}


def get_piecewise_schedule(start_val, end_val, current_step, decay_start, total_steps):
    """
    Implementiert: Konstant bis 'decay_start', danach linearer Abfall.
    """
    if current_step <= decay_start:
        return start_val
    else:
        # Wie weit sind wir im Decay-Prozess? (0.0 bis 1.0)
        steps_in_decay = current_step - decay_start
        total_decay_steps = total_steps - decay_start
        # Safety div by zero
        if total_decay_steps <= 0: return end_val

        progress = steps_in_decay / total_decay_steps
        # Linear Interpolation: Start -> End
        return start_val - (progress * (start_val - end_val))


def train_parallel():
    # --- SETUP ---
    os.makedirs("models_hrl", exist_ok=True)
    os.makedirs("checkpoints_hrl", exist_ok=True)

    plan = Plan()
    plan.duration = 129600
    init_batch = {"time": 0, "number": 10}

    print(
        f">>> Starte Training (Mode: {'IMPLIZIT' if HRL_CONFIG['use_implicit'] else 'EXPLIZIT'}) mit Warmup (Worker={HRL_CONFIG['decay_start_worker']}, Manager={HRL_CONFIG['decay_start_manager']})...")

    # 1. Environment mit Flag starten
    env = SubprocVecEnv(HRL_CONFIG['n_envs'], plan, init_batch, use_implicit=HRL_CONFIG['use_implicit'])

    # --- UPDATE: Dynamische State Dimension ermitteln ---
    # Wir machen den allerersten Reset HIER, um zu sehen wie groß der Vektor wirklich ist!
    raw_states = env.reset()
    state_dim = raw_states.shape[1]  # <- Dies wird automatisch 82 sein!
    print(f"Dynamisch erkannte State-Dimension: {state_dim}")

    # 2. Context Dim berechnen (WICHTIG!)
    context_dim = 6 if HRL_CONFIG['use_implicit'] else 4

    # 3. Netzwerke initialisieren (mit der echten Dimension 82)
    manager = ManagerNetwork(input_dim=state_dim, use_implicit=HRL_CONFIG['use_implicit'])
    worker = WorkerNetwork(state_dim=state_dim, context_dim=context_dim)

    # Optimizer starten mit Start-LR
    opt_manager = optim.Adam(manager.parameters(), lr=HRL_CONFIG['lr_manager_start'])
    opt_worker = optim.Adam(worker.parameters(), lr=HRL_CONFIG['lr_worker_start'])

    norm_manager = RunningMeanStd(shape=(state_dim,))
    norm_worker = RunningMeanStd(shape=(state_dim,))

    f = open("training_log_parallel.csv", "w", newline="")
    writer = csv.writer(f)
    writer.writerow(
        ["Update", "AvgWorkerR", "AvgManagerR", "Profit", "TP", "AvgWIP", "MgrLoss", "LR_W", "Ent_W", "WallTime"])

    start_time = time.time()
    best_avg_profit = -float('inf')

    m_buffer = {'states': [], 'actions': [], 'rewards': [], 'dones': [], 'log_probs': [], 'values': []}

    # --- INITIAL SETUP ---
    norm_manager.update(raw_states)
    norm_worker.update(raw_states)
    states = norm_worker.normalize(raw_states)

    t_states = torch.FloatTensor(norm_manager.normalize(raw_states))
    with torch.no_grad():
        m_dists, m_vals = manager(t_states)

    # Initiale Actions (Dynamisch)
    curr_m_actions_tensor = {
        'capacity': m_dists['capacity'].sample(),
        'release': m_dists['release'].sample(),
        'batch': m_dists['batch'].sample()
    }

    # Zusatz-Keys je nach Modus
    if HRL_CONFIG['use_implicit']:
        curr_m_actions_tensor['prio_low'] = m_dists['prio_low'].sample()
        curr_m_actions_tensor['prio_mid'] = m_dists['prio_mid'].sample()
        curr_m_actions_tensor['prio_high'] = m_dists['prio_high'].sample()
    else:
        curr_m_actions_tensor['strategy'] = m_dists['strategy'].sample()

    # Liste für Environment erstellen
    curr_m_actions_list = []
    for i in range(HRL_CONFIG['n_envs']):
        # Basis
        act = {
            'capacity': curr_m_actions_tensor['capacity'][i].item(),
            'release': curr_m_actions_tensor['release'][i].item(),
            'batch': curr_m_actions_tensor['batch'][i].item()
        }
        # Zusatz
        if HRL_CONFIG['use_implicit']:
            act['prio_low'] = curr_m_actions_tensor['prio_low'][i].item()
            act['prio_mid'] = curr_m_actions_tensor['prio_mid'][i].item()
            act['prio_high'] = curr_m_actions_tensor['prio_high'][i].item()
        else:
            act['strategy'] = curr_m_actions_tensor['strategy'][i].item()

        curr_m_actions_list.append(act)

    m_log_probs = sum([m_dists[k].log_prob(curr_m_actions_tensor[k]) for k in curr_m_actions_tensor])

    m_acc_rewards = np.zeros(HRL_CONFIG['n_envs'])
    m_last_states = t_states.clone()
    m_last_values = m_vals
    m_last_log_probs = m_log_probs
    m_last_actions = {k: v.clone() for k, v in curr_m_actions_tensor.items()}

    ep_profits = []
    ep_throughputs = []
    current_ep_profit = np.zeros(HRL_CONFIG['n_envs'])

    # =========================================================================
    # MAIN LOOP
    # =========================================================================
    for update in range(1, HRL_CONFIG['total_timesteps'] + 1):

        # --- UPDATE SCHEDULING ---
        cur_lr_worker = get_piecewise_schedule(
            HRL_CONFIG['lr_worker_start'], HRL_CONFIG['lr_worker_end'],
            update, HRL_CONFIG['decay_start_worker'], HRL_CONFIG['total_timesteps']
        )
        cur_lr_manager = get_piecewise_schedule(
            HRL_CONFIG['lr_manager_start'], HRL_CONFIG['lr_manager_end'],
            update, HRL_CONFIG['decay_start_manager'], HRL_CONFIG['total_timesteps']
        )
        cur_ent_worker = get_piecewise_schedule(
            HRL_CONFIG['ent_worker_start'], HRL_CONFIG['ent_worker_end'],
            update, HRL_CONFIG['decay_start_worker'], HRL_CONFIG['total_timesteps_math']
        )
        cur_ent_manager = get_piecewise_schedule(
            HRL_CONFIG['ent_manager_start'], HRL_CONFIG['ent_manager_end'],
            update, HRL_CONFIG['decay_start_manager'], HRL_CONFIG['total_timesteps_math']
        )

        for param_group in opt_worker.param_groups:
            param_group['lr'] = cur_lr_worker
        for param_group in opt_manager.param_groups:
            param_group['lr'] = cur_lr_manager

        # ---------------------------------------------------

        w_states, w_contexts, w_actions_list = [], [], []
        w_rewards, w_dones, w_log_probs, w_values = [], [], [], []

        update_wips = []
        update_mgr_rewards = []

        for step in range(HRL_CONFIG['steps_per_update']):

            state_tensor = torch.FloatTensor(states)
            # Context ist curr_m_actions_tensor (enthält bereits die richtigen Keys)
            ctx_input = curr_m_actions_tensor

            with torch.no_grad():
                w_dists, w_vals = worker(state_tensor, ctx_input)

            w_action_tensor = {k: v.sample() for k, v in w_dists.items()}
            w_log_prob = sum([dist.log_prob(w_action_tensor[k]) for k, dist in w_dists.items()])

            w_actions_env_list = []
            for i in range(HRL_CONFIG['n_envs']):
                w_actions_env_list.append({k: v[i].item() for k, v in w_action_tensor.items()})

            raw_next_states, rewards_worker, dones, infos = env.step(w_actions_env_list, curr_m_actions_list)

            norm_manager.update(raw_next_states)
            norm_worker.update(raw_next_states)
            next_states = norm_worker.normalize(raw_next_states)

            rewards_manager = np.array([inf['manager_reward'] for inf in infos])
            wips = [inf.get('wip_count', 0) for inf in infos]
            step_profits = [inf.get('step_profit', 0) for inf in infos]

            current_ep_profit += step_profits
            m_acc_rewards += rewards_manager

            update_wips.extend(wips)
            update_mgr_rewards.extend(rewards_manager)

            w_states.append(state_tensor)
            w_contexts.append({k: v.clone() for k, v in curr_m_actions_tensor.items()})
            w_actions_list.append(w_action_tensor)
            w_rewards.append(torch.FloatTensor(rewards_worker))
            w_dones.append(torch.FloatTensor(dones))
            w_log_probs.append(w_log_prob)
            w_values.append(w_vals.squeeze())

            # Manager Logic
            m_state_now_tensor = torch.FloatTensor(norm_manager.normalize(raw_next_states))

            with torch.no_grad():
                new_m_dists, new_m_vals = manager(m_state_now_tensor)

            # Neue Actions samplen (Dynamisch)
            new_actions = {
                'capacity': new_m_dists['capacity'].sample(),
                'release': new_m_dists['release'].sample(),
                'batch': new_m_dists['batch'].sample()
            }
            if HRL_CONFIG['use_implicit']:
                new_actions['prio_low'] = new_m_dists['prio_low'].sample()
                new_actions['prio_mid'] = new_m_dists['prio_mid'].sample()
                new_actions['prio_high'] = new_m_dists['prio_high'].sample()
            else:
                new_actions['strategy'] = new_m_dists['strategy'].sample()

            new_log_probs = sum([new_m_dists[k].log_prob(new_actions[k]) for k in new_actions])

            for i in range(HRL_CONFIG['n_envs']):
                if infos[i]['manager_needed'] or dones[i]:
                    s_old = m_last_states[i].unsqueeze(0)
                    a_old = {k: v[i].unsqueeze(0) for k, v in m_last_actions.items()}

                    m_buffer['states'].append(s_old)
                    m_buffer['actions'].append(a_old)
                    m_buffer['rewards'].append(m_acc_rewards[i])
                    m_buffer['dones'].append(dones[i])
                    m_buffer['log_probs'].append(m_last_log_probs[i].unsqueeze(0))
                    m_buffer['values'].append(m_last_values[i].unsqueeze(0))

                    m_acc_rewards[i] = 0

                    # Update Environment Inputs
                    for k in curr_m_actions_tensor:
                        curr_m_actions_tensor[k][i] = new_actions[k][i]
                        curr_m_actions_list[i][k] = new_actions[k][i].item()

                    m_last_states[i] = m_state_now_tensor[i]
                    m_last_values[i] = new_m_vals[i]
                    m_last_log_probs[i] = new_log_probs[i]
                    for k in m_last_actions:
                        m_last_actions[k][i] = new_actions[k][i]

                if dones[i]:
                    ep_profits.append(current_ep_profit[i])
                    ep_throughputs.append(infos[i].get('throughput', 0))
                    current_ep_profit[i] = 0

            states = next_states

        # --- UPDATE WORKER ---
        state_tensor = torch.FloatTensor(states)
        with torch.no_grad():
            _, next_vals = worker(state_tensor, curr_m_actions_tensor)

        returns, advantages = calculate_gae_parallel(w_rewards, w_values, w_dones, next_vals.squeeze(), HRL_CONFIG)

        update_policy_worker_parallel(worker, opt_worker, w_states, w_contexts, w_actions_list, w_log_probs, returns,
                                      advantages, HRL_CONFIG, cur_ent_worker)

        # --- UPDATE MANAGER ---
        loss_m_val = 0
        if len(m_buffer['states']) >= 256:
            b_rew = torch.FloatTensor(m_buffer['rewards']).unsqueeze(1)
            b_vals = torch.stack(m_buffer['values']).squeeze(1)
            b_adv = b_rew - b_vals.detach()

            loss_m_val = update_policy_manager_parallel(manager, opt_manager, m_buffer, b_rew, b_adv, HRL_CONFIG,
                                                        cur_ent_manager)

            for k in m_buffer: m_buffer[k] = []

        # --- LOGGING ---
        avg_w_r = np.mean([t.mean().item() for t in w_rewards])
        avg_m_r = np.mean(update_mgr_rewards)
        avg_prof = np.mean(ep_profits[-20:]) if ep_profits else 0
        avg_tp = np.mean(ep_throughputs[-20:]) if ep_throughputs else 0
        avg_wip = np.mean(update_wips) if update_wips else 0

        print(
            f"Upd {update}: WR={avg_w_r:.2f} | MR={avg_m_r:.2f} | Prof={avg_prof / 1000:.1f}k | TP={avg_tp:.0f} | WIP={avg_wip:.1f} | LR={cur_lr_worker:.1e} | Ent={cur_ent_worker:.3f}")
        writer.writerow([update, avg_w_r, avg_m_r, avg_prof, avg_tp, avg_wip, loss_m_val, cur_lr_worker, cur_ent_worker,
                         time.time() - start_time])
        f.flush()

        # Best Model
        if avg_prof > best_avg_profit and update > 100:
            best_avg_profit = avg_prof
            torch.save(manager.state_dict(), "models_hrl/policy_manager_best.pt")
            torch.save(worker.state_dict(), "models_hrl/policy_worker_best.pt")
            np.savez("models_hrl/norm_manager_best.npz", mean=norm_manager.mean, var=norm_manager.var,
                     count=norm_manager.count)
            np.savez("models_hrl/norm_worker_best.npz", mean=norm_worker.mean, var=norm_worker.var,
                     count=norm_worker.count)
            print(f" >>> REKORD! Best Profit: {best_avg_profit / 1000:.1f}k")

        if update > 150 and update % 10 == 0:
            torch.save(manager.state_dict(), f"checkpoints_hrl/manager_par_{update}.pt")
            torch.save(worker.state_dict(), f"checkpoints_hrl/worker_par_{update}.pt")
            np.savez(f"checkpoints_hrl/norm_manager_{update}.npz", mean=norm_manager.mean, var=norm_manager.var,
                     count=norm_manager.count)
            np.savez(f"checkpoints_hrl/norm_worker_{update}.npz", mean=norm_worker.mean, var=norm_worker.var,
                     count=norm_worker.count)

    env.close()
    f.close()
    print("Training Finished.")


# --- HELPER FUNCTIONS ---

def calculate_gae_parallel(rewards, values, dones, next_val, config):
    returns = []
    next_val = next_val
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + config['gamma_worker'] * next_val * (1 - dones[i]) - values[i]
        returns.insert(0, rewards[i] + config['gamma_worker'] * next_val * (1 - dones[i]))
        next_val = values[i]
    return torch.stack(returns), torch.stack(returns) - torch.stack(values)


def update_policy_worker_parallel(net, optimizer, states, contexts, actions, old_log_probs, returns, advantages, config,
                                  ent_coef):
    # --- KORREKTUR: Dynamische State-Dimension auslesen ---
    state_dim = states[0].shape[-1]  # Holt sich automatisch die zz. 82
    b_states = torch.stack(states).view(-1, state_dim)
    # ------------------------------------------------------

    b_returns = returns.view(-1)
    b_advantages = advantages.view(-1)
    b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
    b_old_log_probs = torch.stack(old_log_probs).view(-1)

    # Keys korrekt trennen
    ctx_keys = contexts[0].keys()
    b_contexts = {k: torch.stack([c[k] for c in contexts]).view(-1) for k in ctx_keys}

    act_keys = actions[0].keys()
    b_actions = {k: torch.stack([a[k] for a in actions]).view(-1) for k in act_keys}

    dataset_size = b_states.size(0)
    n_minibatches = 4
    mini_batch_size = max(1, dataset_size // n_minibatches)
    indices = np.arange(dataset_size)

    for _ in range(config['n_epochs']):
        np.random.shuffle(indices)
        for start in range(0, dataset_size, mini_batch_size):
            end = start + mini_batch_size
            idx = indices[start:end]

            mb_states = b_states[idx]
            mb_ctx = {k: v[idx] for k, v in b_contexts.items()}
            mb_actions = {k: v[idx] for k, v in b_actions.items()}
            mb_adv = b_advantages[idx]
            mb_ret = b_returns[idx]
            mb_old_lp = b_old_log_probs[idx]

            dists, vals = net(mb_states, mb_ctx)

            new_lp = sum([dists[k].log_prob(mb_actions[k]) for k in mb_actions])
            entropy = sum([dists[k].entropy() for k in mb_actions]).mean()

            ratio = torch.exp(new_lp - mb_old_lp)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - config['clip_epsilon'], 1.0 + config['clip_epsilon']) * mb_adv

            a_loss = -torch.min(surr1, surr2).mean()
            c_loss = 0.5 * ((vals.squeeze() - mb_ret) ** 2).mean()

            loss = a_loss + c_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), config['max_grad_norm'])
            optimizer.step()


def update_policy_manager_parallel(net, optimizer, buffer, returns, advantages, config, ent_coef):
    b_states = torch.stack(buffer['states']).squeeze(1)
    b_returns = returns.squeeze(1)
    b_adv = advantages.squeeze(1)
    b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

    keys = buffer['actions'][0].keys()
    b_actions = {k: torch.stack([a[k] for a in buffer['actions']]).view(-1) for k in keys}
    b_old_lp = torch.stack(buffer['log_probs']).view(-1)

    for _ in range(config['n_epochs']):
        dists, vals = net(b_states)
        new_lp = sum([dists[k].log_prob(b_actions[k]) for k in b_actions])
        entropy = sum([dists[k].entropy() for k in b_actions]).mean()

        ratio = torch.exp(new_lp - b_old_lp)
        surr1 = ratio * b_adv
        surr2 = torch.clamp(ratio, 1.0 - config['clip_epsilon'], 1.0 + config['clip_epsilon']) * b_adv

        a_loss = -torch.min(surr1, surr2).mean()
        c_loss = 0.5 * ((vals.squeeze() - b_returns) ** 2).mean()

        loss = a_loss + c_loss - ent_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), config['max_grad_norm'])
        optimizer.step()

    return loss.item()


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    train_parallel()