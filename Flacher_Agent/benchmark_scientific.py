# benchmark.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as stats
import simpy
import matplotlib.transforms as mtransforms

# Imports
from learning.simpy_shop_wrapper import SimpyShopWrapper
from learning.policy_ppo import PPOConfig, PolicyNetwork
from learning.action import Action
from learning.utils import RunningMeanStd
from simulation.internal import Plan
from simulation.random_manager import RandomManager

# GA Imports
from meta_heuristics.ga_utils import genome_to_action

# --- KONFIGURATION BENCHMARK & SEEDS ---
NUM_BASE_SEEDS = 10
SEED_START = 1000
SEED_STEP = 1000
EPISODES_PER_SEED = 20

SIM_DURATION = 129600  # 90 Tage
EVAL_IMPLICIT_MODE = True

# Vergleichs-Modi
COMPARE_WITH_GA = True
COMPARE_WITH_TWO = True

# GEWINNER-GENE
BEST_GA_GENES = [3.65264693, 1.34816443, 4.97601367, 4.63929795, 2.91764524, 0.29082064,
                 6.68306114, 3.99347954, 1.55510494, 3.70123454, 4.76335672]

BEST_TWO_GENES = [4.82837385, 0.78102757, 0.00000000, 2.38672926, 3.26666834, 0.22100635,
                  3.41837667, 4.81918683, 2.88577485, 5.35752676, 1.97171123]

# Batch Sizes Liste (muss identisch zum Training sein)
GA_BATCH_SIZES = [5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30]


def plot_single_scientific_ci(results, metric_key, ylabel_math, filename_base):
    """
    Erstellt einen streng wissenschaftlichen Plot (offener Kreis, geschlossener Kasten,
    kein Grid, kein Titel) basierend auf der Referenzoptik (image_4.png).
    Gleichzeitig werden MW und STABW im Terminal ausgegeben.
    """
    names = ['PPO Agent', 'GA', 'TWO']
    # Nur plotten, was auch wirklich ausgewertet wurde
    available_names = [n for n in names if n in results]

    if not available_names:
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    # Konsolen-Ausgabe (wir filtern die Math-Tags grob raus für die Lesbarkeit)
    clean_name = ylabel_math.replace('$', '').replace('\\bar{X}_', 'X-Strich ')
    print(f"\n--- WERTE FÜR: {clean_name} ---")

    # Einheitliche Linienstärke definieren
    lineweedth_val = 2.0

    for i, name in enumerate(available_names):
        data = results[name][metric_key]
        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        se = std / np.sqrt(n)
        ci = 1.96 * se  # 95% Konfidenzintervall

        # Ausgabe im Terminal
        print(f"  {name:15s} | MW = {mean:>10.2f} | STABW = {std:>8.2f} | 95% CI = ±{ci:.2f}")

        # Wissenschaftlicher Style:
        ax.errorbar(x=i, y=mean, yerr=ci, fmt='o', color='black',
                    ecolor='black', elinewidth=lineweedth_val, capsize=6, capthick=1.5,
                    markersize=9, markerfacecolor='white', markeredgecolor='black',
                    markeredgewidth=lineweedth_val, zorder=3)

    # Achsen-Beschriftungen (Mathematische Notation)
    ax.set_ylabel(ylabel_math, fontsize=12)  # Etwas größer, da es eine Formel ist
    ax.set_xlabel("Steuerungsansatz", fontsize=12, labelpad=10)

    ax.set_xticks(range(len(available_names)))
    ax.set_xticklabels(available_names, fontsize=10)

    # Geschlossener Kasten
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.tick_params(top=True, right=True, direction='out')  # Ticks auch oben/rechts, nach außen gerichtet

    # ANPASSUNG 3: Bündigere X-Achsen-Limits für einen geschlossenen Kasten
    n_categories = len(available_names)
    ax.set_xlim(-0.25, n_categories - 0.75)

    # Kein Grid
    ax.grid(False)

    # Y-Achsen-Limits bündiger berechnen
    min_y = float('inf')
    max_y = -float('inf')

    # Suche den tiefsten und höchsten Punkt der CIs
    for n in available_names:
        data = results[n][metric_key]
        n_len = len(data)
        mean = np.mean(data)
        ci = 1.96 * (np.std(data) / np.sqrt(n_len))

        min_y = min(min_y, mean - ci)
        max_y = max(max_y, mean + ci)

    # Spanne berechnen und einen schönen optischen Rand (25%) hinzufügen
    span = max_y - min_y
    if span == 0:
        span = abs(np.mean([max_y, min_y])) * 0.1 + 1.0

    margin = span * 0.25

    # Neues, gezoomtes Limit setzen
    ax.set_ylim(min_y - margin, max_y + margin)

    # Als PDF-Vektorgrafik speichern
    # bbox_inches='tight' schneidet überschüssigen weißen Raum millimetergenau ab
    pdf_filename = f"{filename_base}.pdf"
    plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"-> gespeichert: {pdf_filename}")


def perform_significance_test(results, metric_key, name_a, name_b):
    if name_a not in results or name_b not in results:
        return

    data_a = results[name_a][metric_key]
    data_b = results[name_b][metric_key]

    t_stat, p_val = stats.ttest_ind(data_a, data_b, equal_var=False)

    mean_a = np.mean(data_a)
    mean_b = np.mean(data_b)
    diff_pct = ((mean_a - mean_b) / mean_b) * 100 if mean_b != 0 else 0

    print(f"\n--- T-Test: {metric_key} ---")
    print(f"  {name_a}: {mean_a:,.1f} vs {name_b}: {mean_b:,.1f} | Diff: {diff_pct:+.2f}%")
    if p_val < 0.05:
        print(f"  >> Signifikanter Unterschied! (p = {p_val:.5f})")
    else:
        print(f"  >> Kein signifikanter Unterschied. (p = {p_val:.5f})")


def run_evaluation(policy=None, normalizer=None, name="Agent", mode="agent", seed_sequence=[]):
    total_runs = len(seed_sequence) * EPISODES_PER_SEED
    print(
        f"\n>>> Starte Evaluation für: {name} ({len(seed_sequence)} Seeds á {EPISODES_PER_SEED} Ep. = {total_runs} Runs)")

    plan = Plan()
    plan.duration = SIM_DURATION

    current_implicit_mode = EVAL_IMPLICIT_MODE
    if mode in ["ga", "two"]:
        current_implicit_mode = True

    plan.use_implicit_batch = current_implicit_mode
    init_batch = {"time": 0, "number": 10}
    env = SimpyShopWrapper(plan, init_batch, use_implicit_batch=current_implicit_mode)

    metrics = {
        'profits': [], 'throughputs': [], 'wip_avgs': [],
        'raw_avgs': [], 'done_avgs': [], 'worker_avgs': [],
        'on_time_rates': [], 'late_deviations': []
    }

    static_action = None
    if mode == "ga":
        static_action = genome_to_action(BEST_GA_GENES, GA_BATCH_SIZES)
    elif mode == "two":
        static_action = genome_to_action(BEST_TWO_GENES, GA_BATCH_SIZES)

    for b_seed in seed_sequence:
        for ep in range(EPISODES_PER_SEED):
            raw_state = env.reset()

            current_seed = b_seed + ep
            env.shop.rand_mgr = RandomManager(base_seed=current_seed, config_id=0, run_id=0)

            state = normalizer.normalize(raw_state) if (mode == "agent" and policy and normalizer) else raw_state
            done = False
            ep_profit = 0

            wip_sum, raw_sum, done_sum, worker_sum = 0, 0, 0, 0
            steps = 0

            while not done:
                if mode == "agent" and policy:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    with torch.no_grad():
                        actions_indices, _ = policy(state_tensor, deterministic=True)

                    config = policy.config
                    bs_idx = actions_indices['batch_size'].item()
                    real_bs = config.batch_sizes[bs_idx]

                    strat_idx, bp_low, bp_mid, bp_high = 0, 0, 0, 0
                    if getattr(config, 'use_implicit_batch', False):
                        bp_low = actions_indices.get('batch_prio_low', torch.tensor(0)).item()
                        bp_mid = actions_indices.get('batch_prio_mid', torch.tensor(0)).item()
                        bp_high = actions_indices.get('batch_prio_high', torch.tensor(0)).item()
                    else:
                        strat_idx = actions_indices.get('batch_strategy', torch.tensor(0)).item()

                    action = Action(
                        prio_disassembly=actions_indices['prio_disassembly'].item(),
                        prio_inspection=actions_indices['prio_inspection'].item(),
                        prio_cleaning=actions_indices['prio_cleaning'].item(),
                        prio_repair=actions_indices['prio_repair'].item(),
                        prio_assembly=actions_indices['prio_assembly'].item(),
                        order_release=actions_indices['order_release'].item(),
                        batch_size=real_bs,
                        capacity_level=actions_indices['capacity_level'].item(),
                        batch_strategy=strat_idx,
                        batch_prio_low=bp_low,
                        batch_prio_mid=bp_mid,
                        batch_prio_high=bp_high
                    )
                else:
                    action = static_action

                raw_next_state, reward, done, info = env.step(action)
                state = normalizer.normalize(raw_next_state) if (
                            mode == "agent" and policy and normalizer) else raw_next_state

                ep_profit += info.get('step_profit', 0)
                wip_sum += info.get('wip_count', 0)
                worker_sum += info.get('workers', 20)

                curr_raw = len(env.shop.buffer_q0.items) + len(env.shop.buffer_q1.items) + len(env.shop.buffer_q2.items) + len(env.shop.core_queue.items)
                raw_sum += curr_raw
                done_sum += len(env.shop.product_done.items)

                steps += 1

                if done:
                    if hasattr(env.shop, 'tardiness_history') and len(env.shop.tardiness_history) > 0:
                        tardiness = np.array(env.shop.tardiness_history)
                        on_time_pct = np.mean(tardiness <= 0) * 100.0
                        late_only = tardiness[tardiness > 0]
                        mean_late_days = np.mean(late_only) / (24 * 60) if len(late_only) > 0 else 0.0
                    else:
                        on_time_pct = 0.0
                        mean_late_days = 0.0

                    metrics['on_time_rates'].append(on_time_pct)
                    metrics['late_deviations'].append(mean_late_days)

            # Post-Episode aggregations
            metrics['profits'].append(ep_profit)
            metrics['throughputs'].append(info['throughput'])
            metrics['wip_avgs'].append(wip_sum / steps if steps else 0)
            metrics['raw_avgs'].append(raw_sum / steps if steps else 0)
            metrics['done_avgs'].append(done_sum / steps if steps else 0)
            metrics['worker_avgs'].append(worker_sum / steps if steps else 20)

        print(f"  [Base Seed {b_seed}] {EPISODES_PER_SEED} Episoden verarbeitet.")

    return metrics


def main():
    dummy_plan = Plan()
    dummy_env = SimpyShopWrapper(dummy_plan, {"time": 0, "number": 10}, use_implicit_batch=EVAL_IMPLICIT_MODE)
    dummy_state = dummy_env.reset()
    detected_dim = len(dummy_state)
    del dummy_env

    config = PPOConfig(input_dim=detected_dim, use_implicit_batch=EVAL_IMPLICIT_MODE)
    model_path = os.path.join("models", "policy_step_383.pt")
    norm_path = os.path.join("models", "normalizer_step_383.npz")

    policy, normalizer = None, None
    if os.path.exists(model_path):
        policy = PolicyNetwork(config)
        policy.load_state_dict(torch.load(model_path))
        policy.eval()
        print("Agent geladen.")

    if os.path.exists(norm_path):
        normalizer = RunningMeanStd(shape=(detected_dim,))
        data = np.load(norm_path)
        normalizer.mean = data['mean']
        normalizer.var = data['var']
        print("Normalizer geladen.")

    # Seed-Sequenz für 75 Seeds generieren
    seed_sequence = [SEED_START + (i * SEED_STEP) for i in range(NUM_BASE_SEEDS)]
    results = {}

    if policy:
        results['PPO Agent'] = run_evaluation(policy, normalizer, "PPO Agent", "agent", seed_sequence)
    if COMPARE_WITH_GA:
        results['GA'] = run_evaluation(None, None, "GA Heuristik", "ga", seed_sequence)
    if COMPARE_WITH_TWO:
        results['TWO'] = run_evaluation(None, None, "TWO Heuristik", "two", seed_sequence)

    print("\nGeneriere CI-Plots pro Metrik...")

    # --- PRIMÄR-METRIKEN ---
    plot_single_scientific_ci(results, 'wip_avgs',
                              r"$\bar{X}_{WIP}$ [Teile]", 'eval_metric_wip')

    plot_single_scientific_ci(results, 'throughputs',r"$\bar{X}_{Durchsatz}$ [Stück]", 'eval_metric_throughput')
    plot_single_scientific_ci(results, 'on_time_rates',r"$\bar{X}_{Liefertreue}$ [%]", 'eval_metric_delivery_pct')
    plot_single_scientific_ci(results, 'late_deviations',r"$\bar{X}_{Verspätung}$ [Tage]", 'eval_metric_delivery_days')
    plot_single_scientific_ci(results, 'raw_avgs',r"$\bar{X}_{Bestand-Eingangslager}$ [Cores]", 'eval_metric_raw_inv')
    plot_single_scientific_ci(results, 'done_avgs',r"$\bar{X}_{Bestand-Ausgangslager}$ [Cores]", 'eval_metric_done_inv')

    # --- SEKUNDÄR-METRIKEN ---
    plot_single_scientific_ci(results, 'profits',r"$\bar{X}_{Profit}$ [€]", 'eval_metric_profit')
    plot_single_scientific_ci(results, 'worker_avgs',r"$\bar{X}_{Arbeiter}$ [Personen]", 'eval_metric_workers')

    print("\n" + "=" * 40)
    print("STATISTISCHE ANALYSE (T-Test)")
    print("=" * 40)
    for target in ['PPO Agent', 'GA', 'TWO']:
        if target != 'PPO Agent' and 'PPO Agent' in results and target in results:
            print(f"\n>>>> VERGLEICH: PPO vs {target}")
            perform_significance_test(results, 'profits', 'PPO Agent', target)
            perform_significance_test(results, 'throughputs', 'PPO Agent', target)
            perform_significance_test(results, 'wip_avgs', 'PPO Agent', target)


if __name__ == "__main__":
    main()