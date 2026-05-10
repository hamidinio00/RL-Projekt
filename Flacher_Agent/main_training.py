# learning/main_training.py
import torch
import torch.optim as optim
import numpy as np
import csv
import time
import os
import multiprocessing as mp
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ConstantLR
from torch.utils.data import BatchSampler, SubsetRandomSampler

from learning.simpy_shop_wrapper import SimpyShopWrapper
from simulation.internal import Plan
from learning.vec_env import SubprocVecEnv
from learning.policy_ppo import PPOConfig, PolicyNetwork
from learning.action import Action
from learning.utils import RunningMeanStd


def train_parallel():
    # --- Setup ---
    os.makedirs("models", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    n_envs = 6
    print(f">>> Starte Paralleles Training auf {n_envs} Prozessen...")

    IMPLICIT_MODE = True

    # Plan Setup
    plan = Plan()
    plan.duration = 129600  # ca. 3 Monate Simulation
    plan.use_implicit_batch = IMPLICIT_MODE
    init_batch = {"time": 0, "number": 10}

    # --- 1. AUTO-DETECT STATE DIM (KRITISCHER FIX) ---
    # Wir starten eine Dummy-Umgebung, nur um die Größe des State-Vektors zu messen.
    print("Messe State-Dimension...")
    dummy_env = SimpyShopWrapper(plan, init_batch, use_implicit_batch=IMPLICIT_MODE)
    dummy_state = dummy_env.reset()
    actual_state_dim = len(dummy_state)
    print(f">>> State Dimension erkannt: {actual_state_dim}")
    # Dummy schließen (nicht mehr benötigt)
    del dummy_env

    # --- 2. Config mit korrekter Dimension ---
    config = PPOConfig(input_dim=actual_state_dim, use_implicit_batch=IMPLICIT_MODE)

    # --- 3. Vector Environment starten ---
    vec_env = SubprocVecEnv(plan, init_batches=init_batch, n_envs=n_envs, use_implicit=IMPLICIT_MODE)

    policy = PolicyNetwork(config)
    optimizer = optim.Adam(policy.parameters(), lr=config.lr_actor)

    # Hyperparameter (Optimiert für Langläufer)
    num_updates = 400
    total_timesteps_math = 500
    steps_per_update = 256
    batch_size = n_envs * steps_per_update  # 1536
    mini_batch_size = batch_size // 4  # 384
    n_epochs = 4

    # Scheduler
    scheduler1 = ConstantLR(optimizer, factor=1.0, total_iters=150)
    scheduler2 = LinearLR(optimizer, start_factor=1.0, end_factor=0.2, total_iters=350)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[150])

    # Normalizer mit korrekter Dimension
    state_normalizer = RunningMeanStd(shape=(actual_state_dim,))

    # Logging
    log_filename = "training_log_parallel.csv"
    f = open(log_filename, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(
        ["Update", "AvgReward", "AvgProfit", "AvgThroughput", "AvgWIP", "Loss", "AvgMix", "LR", "EntCoef", "WallTime"])

    # Init
    raw_states = vec_env.reset()
    state_normalizer.update(raw_states)
    states = state_normalizer.normalize(raw_states)

    current_ep_profits = np.zeros(n_envs)
    completed_profits = []
    completed_throughputs = []

    start_time_global = time.time()

    # --- Training Loop ---
    for update in range(1, num_updates + 1):

        b_states, b_actions_idxs, b_rewards, b_log_probs, b_dones, b_values = [], [], [], [], [], []

        # Stats Buffer für dieses Update
        all_mix_ratios_update = []
        all_wip_counts_update = []

        # --- A. Rollout Phase ---
        for step in range(steps_per_update):
            state_tensor = torch.FloatTensor(states)

            with torch.no_grad():
                action_indices, values = policy(state_tensor)
                log_probs, _, _ = policy.evaluate(state_tensor, action_indices)

            # Action Object Construction
            actions_list = []
            for i in range(n_envs):
                bs_idx = action_indices['batch_size'][i].item()
                real_bs = config.batch_sizes[bs_idx]

                strat_idx = 0
                bp_low = 0
                bp_mid = 0
                bp_high = 0

                if config.use_implicit_batch:
                    bp_low = action_indices['batch_prio_low'][i].item()
                    bp_mid = action_indices['batch_prio_mid'][i].item()
                    bp_high = action_indices['batch_prio_high'][i].item()
                else:
                    if 'batch_strategy' in action_indices:
                        strat_idx = action_indices['batch_strategy'][i].item()

                act = Action(
                    prio_disassembly=action_indices['prio_disassembly'][i].item(),
                    prio_inspection=action_indices['prio_inspection'][i].item(),
                    prio_cleaning=action_indices['prio_cleaning'][i].item(),
                    prio_repair=action_indices['prio_repair'][i].item(),
                    prio_assembly=action_indices['prio_assembly'][i].item(),
                    order_release=action_indices['order_release'][i].item(),
                    batch_size=real_bs,
                    capacity_level=action_indices['capacity_level'][i].item(),
                    batch_strategy=strat_idx,
                    batch_prio_low=bp_low,
                    batch_prio_mid=bp_mid,
                    batch_prio_high=bp_high
                )
                actions_list.append(act)

            # Env Step
            raw_next_states, rewards, dones, infos = vec_env.step(actions_list)

            # Normalization
            state_normalizer.update(raw_next_states)
            next_states = state_normalizer.normalize(raw_next_states)

            # Tracking
            for i in range(n_envs):
                profit = infos[i].get('step_profit', 0.0)
                current_ep_profits[i] += profit

                mix = infos[i].get('mix_ratio', 0.0)
                wip = infos[i].get('wip_count', 0)
                all_mix_ratios_update.append(mix)
                all_wip_counts_update.append(wip)

                if dones[i]:
                    completed_profits.append(current_ep_profits[i])
                    current_ep_profits[i] = 0.0
                    tp = infos[i].get('throughput', 0)
                    completed_throughputs.append(tp)

            b_states.append(state_tensor)
            b_actions_idxs.append(action_indices)
            b_rewards.append(torch.FloatTensor(rewards))
            b_dones.append(torch.FloatTensor(dones))
            b_log_probs.append(log_probs.detach())
            b_values.append(values.detach().squeeze())

            states = next_states

        # --- B. GAE Calculation ---
        with torch.no_grad():
            next_val = policy.critic_head(policy.shared_net(torch.FloatTensor(next_states))).squeeze()

        returns = torch.zeros_like(torch.stack(b_rewards))
        advantages = torch.zeros_like(torch.stack(b_rewards))

        gae = 0
        for t in reversed(range(steps_per_update)):
            if t == steps_per_update - 1:
                nextnonterminal = 1.0 - torch.FloatTensor(dones)
                nextvalues = next_val
            else:
                nextnonterminal = 1.0 - b_dones[t + 1]
                nextvalues = b_values[t + 1]

            delta = b_rewards[t] + config.gamma * nextvalues * nextnonterminal - b_values[t]
            gae = delta + config.gamma * config.gae_lambda * nextnonterminal * gae
            advantages[t] = gae
            returns[t] = gae + b_values[t]

        flat_states = torch.cat(b_states)
        flat_log_probs = torch.cat(b_log_probs)
        flat_returns = returns.view(-1)
        flat_advantages = advantages.view(-1)

        flat_actions = {}
        for key in action_indices.keys():
            flat_actions[key] = torch.cat([step_dict[key] for step_dict in b_actions_idxs])

        # --- C. Update ---
        # Normalize Advantages
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

        # Dynamic Entropy
        start_entropy = 0.03
        end_entropy = 0.003
        decay_start = 150

        if update <= decay_start:
            current_ent_coef = start_entropy
        else:
            progress = (update - decay_start) / (total_timesteps_math - decay_start)
            current_ent_coef = max(end_entropy, start_entropy - (progress * (start_entropy - end_entropy)))

        idxs = range(batch_size)
        for _ in range(n_epochs):
            sampler = BatchSampler(SubsetRandomSampler(idxs), mini_batch_size, drop_last=False)
            for mb_idxs in sampler:
                mb_states = flat_states[mb_idxs]
                mb_log_probs = flat_log_probs[mb_idxs]
                mb_advantages = flat_advantages[mb_idxs]
                mb_returns = flat_returns[mb_idxs]
                mb_actions = {k: v[mb_idxs] for k, v in flat_actions.items()}

                new_log_probs, new_values, entropy = policy.evaluate(mb_states, mb_actions)
                ratios = torch.exp(new_log_probs - mb_log_probs)

                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon) * mb_advantages
                v_loss = 0.5 * ((new_values.squeeze() - mb_returns) ** 2).mean()
                loss = -torch.min(surr1, surr2).mean() + v_loss - current_ent_coef * entropy.mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
                optimizer.step()

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # --- D. Logging ---
        elapsed = time.time() - start_time_global
        avg_rew = flat_returns.mean().item()
        # Nur Profit von fertigen Episoden nehmen für Glättung
        avg_prof = np.mean(completed_profits[-100:]) if completed_profits else 0.0
        avg_tp = np.mean(completed_throughputs[-100:]) if completed_throughputs else 0.0
        avg_mix = np.mean(all_mix_ratios_update) if all_mix_ratios_update else 0.0
        avg_wip = np.mean(all_wip_counts_update) if all_wip_counts_update else 0.0

        print(
            f"Upd {update}: R={avg_rew:.2f} | Prof={avg_prof / 1000:.1f}k | TP={avg_tp:.1f} | WIP={avg_wip:.0f} | Ent={current_ent_coef:.4f}")

        writer.writerow(
            [update, avg_rew, avg_prof, avg_tp, avg_wip, loss.item(), avg_mix, current_lr, current_ent_coef, elapsed])

        if update >= 200 and update % 1 == 0:
            torch.save(policy.state_dict(), f"checkpoints/policy_step_{update}.pt")
            np.savez(f"checkpoints/normalizer_step_{update}.npz", mean=state_normalizer.mean, var=state_normalizer.var)

    vec_env.close()
    f.close()
    print("Training Finished.")

    # --- Plotting: 4 Subplots ---
    try:
        import pandas as pd
        import matplotlib.pyplot as plt

        df = pd.read_csv("training_log_parallel.csv")

        # JETZT 4 Subplots untereinander (Höhe angepasst auf 20)
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 20), sharex=True)

        # Plot 1: Profit & Reward
        color = 'tab:blue'
        ax1.set_ylabel('Avg Profit (€)', color=color, fontweight='bold')
        ax1.plot(df['Update'], df['AvgProfit'], color=color, label='Profit')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)
        ax1.set_title("Training Metrics Overview")

        # Twin Axis für Reward im ersten Plot
        ax1b = ax1.twinx()
        color = 'tab:red'
        ax1b.set_ylabel('Avg Reward', color=color)
        ax1b.plot(df['Update'], df['AvgReward'], color=color, alpha=0.3, linestyle='--', label='Reward')
        ax1b.tick_params(axis='y', labelcolor=color)

        # Plot 2: Throughput
        color = 'tab:green'
        ax2.set_ylabel('Avg Throughput', color=color, fontweight='bold')
        ax2.plot(df['Update'], df['AvgThroughput'], color=color, label='Throughput')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.grid(True, alpha=0.3)

        # Plot 3: WIP (Work in Process)
        color = 'tab:orange'
        ax3.set_ylabel('Avg WIP Count', color=color, fontweight='bold')
        ax3.plot(df['Update'], df['AvgWIP'], color=color, label='WIP')
        ax3.tick_params(axis='y', labelcolor=color)
        ax3.grid(True, alpha=0.3)

        # Plot 4: Mix Ratio
        color = 'tab:purple'
        ax4.set_xlabel('Update Step')
        ax4.set_ylabel('Avg Mix Ratio (0-1)', color=color, fontweight='bold')
        ax4.plot(df['Update'], df['AvgMix'], color=color, label='Mix Ratio')
        ax4.tick_params(axis='y', labelcolor=color)
        ax4.set_ylim(-0.05, 1.05)  # Festlegen auf 0-100%
        ax4.grid(True, alpha=0.3)

        fig.tight_layout()
        plt.savefig("training_results.png")  # Speichern als Bild
        plt.show()
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    train_parallel()