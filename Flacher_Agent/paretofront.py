import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import os
import pandas as pd

# Imports deiner Umgebung
from learning.simpy_shop_wrapper import SimpyShopWrapper
from learning.policy_ppo import PPOConfig, PolicyNetwork
from learning.utils import RunningMeanStd
from simulation.internal import Plan
from simulation.random_manager import RandomManager

# --- CONFIG ---
CHECKPOINT_DIR = "checkpoints"
EVAL_EPISODES = 50
SIM_DURATION = 129600
IMPLICIT_MODE = True


def get_action_from_policy(idx, config):
    from learning.action import Action
    real_bs = config.batch_sizes[idx['batch_size'].item()]
    strat = 0
    bp_l, bp_m, bp_h = 0, 0, 0

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

def evaluate_model(policy_path, norm_path):
    plan = Plan()
    plan.duration = SIM_DURATION
    plan.use_implicit_batch = IMPLICIT_MODE

    env = SimpyShopWrapper(plan, {"time": 0, "number": 10}, use_implicit_batch=IMPLICIT_MODE)
    dummy_state = env.reset()
    input_dim = len(dummy_state)

    config = PPOConfig(input_dim=input_dim, use_implicit_batch=IMPLICIT_MODE)
    policy = PolicyNetwork(config)
    try:
        policy.load_state_dict(torch.load(policy_path))
        policy.eval()
    except Exception as e:
        print(f"Fehler bei {policy_path}: {e}")
        return None

    normalizer = RunningMeanStd(shape=(input_dim,))
    try:
        data = np.load(norm_path)
        normalizer.mean = data['mean']
        normalizer.var = data['var']
    except:
        return None

    profits = []
    wips = []
    throughputs = []

    # WICHTIG: Reset VOR Seed setzen!
    seeds = [1000 + i for i in range(EVAL_EPISODES)]

    for seed in seeds:
        # 1. Seed direkt im Environment setzen
        env.base_seed = seed

        # 2. Reset aufrufen (der Wrapper nutzt nun intern den neuen base_seed)
        raw_state = env.reset()

        state = normalizer.normalize(raw_state)
        done = False

        ep_profit = 0
        wip_sum = 0
        steps = 0

        while not done:
            with torch.no_grad():
                idx, _ = policy(torch.FloatTensor(state).unsqueeze(0), deterministic=True)

            action = get_action_from_policy(idx, config)
            raw_next, _, done, info = env.step(action)
            state = normalizer.normalize(raw_next)

            ep_profit += info.get('step_profit', 0)
            wip_sum += info.get('wip_count', 0)
            steps += 1

        profits.append(ep_profit)
        wips.append(wip_sum / max(1, steps))
        throughputs.append(env.shop.total_throughput)

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

    # Suche nach .pt Dateien
    files = glob.glob(os.path.join(CHECKPOINT_DIR, "*.pt"))
    files = sorted(files, key=os.path.getmtime)

    if not files:
        print("KEINE Policy-Dateien (.pt) gefunden.")
        return

    data = []
    print(f"Gefunden: {len(files)} Modelle. Starte Evaluation...")

    for i, p_path in enumerate(files):
        base_name = os.path.basename(p_path)  # z.B. policy_step_100.pt

        # Extrahiere Nummer (Robust)
        # Wir entfernen alles was nicht Zahl ist
        step_str = ''.join(filter(str.isdigit, base_name))

        if not step_str:
            print(f"Skipping {base_name}: Keine Nummer im Dateinamen.")
            continue

        # Wir probieren verschiedene Namensschemata für den Normalizer
        possible_norms = [
            f"norm_step_{step_str}.npz",  # Schema Pareto
            f"normalizer_step_{step_str}.npz",  # Schema Main
            f"normalizer_{step_str}.npz",  # Schema Alt
            f"norm_{step_str}.npz"
        ]

        found_norm = False
        n_path = ""

        for name in possible_norms:
            test_path = os.path.join(CHECKPOINT_DIR, name)
            if os.path.exists(test_path):
                n_path = test_path
                found_norm = True
                break

        if found_norm:
            print(f"[{i + 1}/{len(files)}] Evaluiere Step {step_str}...", end="\r")
            res = evaluate_model(p_path, n_path)
            if res:
                res['step'] = int(step_str)
                data.append(res)
        else:
            # Nur Warnung bei den ersten paar, sonst spam
            if i < 3:
                print(f"\nWARNUNG: Kein Normalizer gefunden für {base_name}!")
                print(f"   Gesucht: {possible_norms}")

    if not data:
        print("\n\nFEHLER: Keine Modelle konnten evaluiert werden (Datenliste leer).")
        print("Prüfe ob policy_*.pt und norm_*.npz Dateien zusammenpassen.")
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
    print("PARETO OPTIMALE LÖSUNGEN")
    print("=" * 60)
    if not df_pareto.empty:
        print(df_pareto[['name', 'profit', 'throughput', 'wip']].sort_values(by='profit', ascending=False).to_string(
            index=False))
    else:
        print("Keine Pareto-Lösungen gefunden (unwahrscheinlich).")

    df.to_csv("pareto_analysis_results.csv", index=False)
    print("\nDaten gespeichert in 'pareto_analysis_results.csv'")

    # --- PLOTTING (3D) ---
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(df['wip'], df['throughput'], df['profit'], c='gray', alpha=0.3, label='Dominierte Modelle')

    if not df_pareto.empty:
        p = ax.scatter(df_pareto['wip'], df_pareto['throughput'], df_pareto['profit'],
                       c=df_pareto['step'], cmap='viridis', s=100, label='Pareto Front', depthshade=False)
        cbar = plt.colorbar(p, ax=ax, pad=0.1)
        cbar.set_label('Training Step')

    # Achsen mit statistischen Formeln beschriften
    ax.set_xlabel(r'$\bar{X}_{WIP}$ [Stück]')
    ax.set_ylabel(r'$\bar{X}_{Durchsatz}$ [Stück]')
    ax.set_zlabel(r'$\bar{X}_{Profit}$ [€]')

    ax.set_title('Pareto-Front-Analyse')
    plt.legend()
    plt.tight_layout()
    plt.savefig("pareto_front_3d.png")

    # --- PLOTTING (2D) ---
    plt.figure(figsize=(10, 6))

    # Graue Punkte für dominierte Modelle
    plt.scatter(df['wip'], df['throughput'], c='gray', alpha=0.4, label='Dominierte Modelle')

    if not df_pareto.empty:
        # Rote Punkte für Pareto-Lösungen
        plt.scatter(df_pareto['wip'], df_pareto['throughput'], c='red', s=80, marker='o', label='Pareto Front')
        for _, row in df_pareto.iterrows():
            plt.text(row['wip'] + 0.5, row['throughput'] + 0.5, f"{int(row['step'])}", fontsize=8)

    # Achsen mit statistischen Formeln beschriften
    plt.xlabel(r'$\bar{X}_{WIP}$ [Stück]')
    plt.ylabel(r'$\bar{X}_{Durchsatz}$ [Stück]')
    plt.title("Pareto-Front-Analyse")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pareto_front_2d.png")

    plt.show()


if __name__ == "__main__":
    main()