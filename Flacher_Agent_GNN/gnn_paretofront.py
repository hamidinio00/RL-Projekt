import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import os
import pandas as pd
from torch_geometric.data import Batch

# Imports aus deinem Projekt
from simulation.internal import Plan
from learning.action import Action
from learning.simpy_shop_wrapper import SimpyShopWrapper
from simulation.random_manager import RandomManager

# GNN spezifische Imports
from learning_GNN.gnn_wrapper import GraphObservationWrapper
from learning_GNN.gnn_agent import GNNAgent, PPOConfigGNN

# --- CONFIG ---
# ACHTUNG: Pfad anpassen, falls deine GNN-Modelle woanders liegen!
CHECKPOINT_DIR = "models_gnn"
EVAL_EPISODES = 20
SIM_DURATION = 129600
IMPLICIT_MODE = True


def get_action_from_policy(idx, config):
    real_bs = config.batch_sizes[idx['batch_size'].item()]
    strat = 0
    bp_l, bp_m, bp_h = 0.0, 0.0, 0.0

    if getattr(config, 'use_implicit_batch', False):
        bp_l = idx['batch_prio_low'].item()
        bp_m = idx['batch_prio_mid'].item()
        bp_h = idx['batch_prio_high'].item()
    elif 'batch_strategy' in idx:
        strat = idx['batch_strategy'].item()

    return Action(
        prio_disassembly=idx['prio_disassembly'].item(),
        prio_inspection=idx['prio_inspection'].item(),
        prio_cleaning=idx['prio_cleaning'].item(),
        prio_repair=idx['prio_repair'].item(),
        prio_assembly=idx['prio_assembly'].item(),
        order_release=idx['order_release'].item(),
        batch_size=real_bs,
        capacity_level=idx['capacity_level'].item(),
        batch_strategy=strat,
        batch_prio_low=bp_l, batch_prio_mid=bp_m, batch_prio_high=bp_h
    )


def evaluate_model(policy_path):
    plan = Plan()
    plan.duration = SIM_DURATION
    plan.use_implicit_batch = IMPLICIT_MODE

    # SimPy Wrapper initialisieren
    sim_env = SimpyShopWrapper(plan, {"time": 0, "number": 10}, use_implicit_batch=IMPLICIT_MODE)

    # NEU: GNN Wrapper darüberlegen
    env = GraphObservationWrapper(sim_env)

    config = PPOConfigGNN(use_implicit_batch=IMPLICIT_MODE)
    policy = GNNAgent(config)
    try:
        policy.load_state_dict(torch.load(policy_path))
        policy.eval()
    except Exception as e:
        print(f"Fehler bei {policy_path}: {e}")
        return None

    # WICHTIG: Das GNN braucht KEINEN RunningMeanStd Normalizer,
    # da die Features (Ratios, Alters-Norm, One-Hot) bereits im Wrapper auf 0-1 normiert werden!

    profits = []
    wips = []
    throughputs = []

    seeds = [1000 + i for i in range(EVAL_EPISODES)]

    for seed in seeds:
        # Seed direkt im darunterliegenden SimPy-Env setzen
        env.env.base_seed = seed

        # Reset aufrufen
        state_graph, _ = env.reset()
        done = False

        ep_profit = 0
        wip_sum = 0
        steps = 0

        while not done:
            # GNN erwartet einen PyG Batch!
            batch_data = Batch.from_data_list([state_graph])

            with torch.no_grad():
                idx, _ = policy(batch_data, deterministic=True)

            action = get_action_from_policy(idx, config)

            # Step liefert nun (obs, reward, terminated, truncated, info) zurück
            state_graph, reward, done, truncated, info = env.step(action)

            # Wenn die Simulation abbricht
            if truncated:
                done = True

            ep_profit += info.get('step_profit', 0)
            wip_sum += info.get('wip_count', 0)
            steps += 1

        profits.append(ep_profit)
        wips.append(wip_sum / max(1, steps))
        throughputs.append(env.env.shop.total_throughput)

    return {
        "name": os.path.basename(policy_path),
        "profit": np.mean(profits),
        "wip": np.mean(wips),
        "throughput": np.mean(throughputs)
    }


def is_dominated(candidate, others):
    c_prof = candidate['profit']
    c_tp = candidate['throughput']
    c_wip = candidate['wip']

    for other in others:
        if other['name'] == candidate['name']:
            continue

        o_prof = other['profit']
        o_tp = other['throughput']
        o_wip = other['wip']

        # Dominanz: Anderer ist BESSER/GLEICH in allem UND strikt BESSER in einem
        # Profit: MAX, TP: MAX, WIP: MIN
        if (o_prof >= c_prof and o_tp >= c_tp and o_wip <= c_wip):
            if (o_prof > c_prof or o_tp > c_tp or o_wip < c_wip):
                return True

    return False


def main():
    print(f"Suche Checkpoints in '{CHECKPOINT_DIR}'...")
    if not os.path.exists(CHECKPOINT_DIR):
        print(f"FEHLER: Ordner '{CHECKPOINT_DIR}' existiert nicht!")
        return

    # Suche nach .pt Dateien (GNN Modelle)
    files = glob.glob(os.path.join(CHECKPOINT_DIR, "*.pt"))
    files = sorted(files, key=os.path.getmtime)

    if not files:
        print("KEINE Policy-Dateien (.pt) gefunden.")
        return

    data = []
    print(f"Gefunden: {len(files)} Modelle. Starte Evaluation...")

    for i, p_path in enumerate(files):
        base_name = os.path.basename(p_path)

        # "best_profit.pt" etc. überspringen, da sie keine Step-Nummer haben
        if "best" in base_name or "last" in base_name:
            continue

        step_str = ''.join(filter(str.isdigit, base_name))

        if not step_str:
            continue

        print(f"[{i + 1}/{len(files)}] Evaluiere GNN Step {step_str}...", end="\r")

        # NEU: Wir brauchen keinen Normalizer-Pfad mehr übergeben!
        res = evaluate_model(p_path)
        if res:
            res['step'] = int(step_str)
            data.append(res)

    if not data:
        print("\n\nFEHLER: Keine Modelle konnten evaluiert werden.")
        return

    print("\n\nBerechne Pareto-Front...")

    pareto_front = []
    for cand in data:
        if not is_dominated(cand, data):
            cand['pareto'] = True
            pareto_front.append(cand)
        else:
            cand['pareto'] = False

    df = pd.DataFrame(data)
    df_pareto = pd.DataFrame(pareto_front)

    print("\n" + "=" * 60)
    print("PARETO OPTIMALE LÖSUNGEN (GNN)")
    print("=" * 60)
    if not df_pareto.empty:
        print(df_pareto[['name', 'profit', 'throughput', 'wip']].sort_values(by='profit', ascending=False).to_string(
            index=False))
    else:
        print("Keine Pareto-Lösungen gefunden.")

    df.to_csv("pareto_gnn_results.csv", index=False)
    print("\nDaten gespeichert in 'pareto_gnn_results.csv'")

    # --- PLOTTING (3D) ---
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(df['wip'], df['throughput'], df['profit'], c='gray', alpha=0.3, label='Dominierte Modelle')

    if not df_pareto.empty:
        p = ax.scatter(df_pareto['wip'], df_pareto['throughput'], df_pareto['profit'],
                       c=df_pareto['step'], cmap='viridis', s=100, label='Pareto Front', depthshade=False)
        cbar = plt.colorbar(p, ax=ax, pad=0.1)
        cbar.set_label('Training Step')

    ax.set_xlabel(r'$\bar{X}_{WIP}$ [Stück]')
    ax.set_ylabel(r'$\bar{X}_{Durchsatz}$ [Stück]')
    ax.set_zlabel(r'$\bar{X}_{Profit}$ [€]')

    ax.set_title('Pareto-Front-Analyse (GNN)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("pareto_gnn_3d.png")

    # --- PLOTTING (2D) ---
    plt.figure(figsize=(10, 6))

    plt.scatter(df['wip'], df['throughput'], c='gray', alpha=0.4, label='Dominierte Modelle')

    if not df_pareto.empty:
        plt.scatter(df_pareto['wip'], df_pareto['throughput'], c='red', s=80, marker='o', label='Pareto Front')
        for _, row in df_pareto.iterrows():
            plt.text(row['wip'] + 0.5, row['throughput'] + 0.5, f"{int(row['step'])}", fontsize=8)

    plt.xlabel(r'$\bar{X}_{WIP}$ [Stück]')
    plt.ylabel(r'$\bar{X}_{Durchsatz}$ [Stück]')
    plt.title("Pareto-Front-Analyse (GNN)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pareto_gnn_2d.png")

    plt.show()


if __name__ == "__main__":
    main()