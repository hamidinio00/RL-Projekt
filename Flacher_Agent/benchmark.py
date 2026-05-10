# benchmark.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as stats
import simpy

# Imports
from learning.simpy_shop_wrapper import SimpyShopWrapper
from learning.policy_ppo import PPOConfig, PolicyNetwork
from learning.action import Action
from learning.utils import RunningMeanStd
from simulation.internal import Plan
from simulation.random_manager import RandomManager

# GA Imports
from meta_heuristics.ga_utils import genome_to_action

# --- KONFIGURATION ---
EVAL_IMPLICIT_MODE = True  # Muss zum trainierten Agenten passen!

SIM_DURATION = 129600  # 90 Tage
DAYS_TO_PLOT = SIM_DURATION // 1440

# Vergleichs-Modi
COMPARE_WITH_RANDOM = False
COMPARE_WITH_GA = True
COMPARE_WITH_TWO = True  # Schalte dies an, wenn du TWO Gene hast

# GEWINNER-GENE
# [Prio1, Prio2, Prio3, Prio4, Prio5, Release, BatchIdx, Cap, W1, W2, W3]
BEST_GA_GENES = [4.07900316, 1.04875883, 1.66963809, 2.11000081, 4.39675409, 0.27512872,
                 5.53971119, 1.2801863,  3.58501314, 3.94301513, 6.62912036]

# GEWINNER-GENE
BEST_TWO_GENES = [4.82837385, 0.78102757, 0.00000000, 2.38672926, 3.26666834, 0.22100635,
                  3.41837667, 4.81918683, 2.88577485, 5.35752676, 1.97171123]

# Batch Sizes Liste (muss identisch zum Training sein)
GA_BATCH_SIZES = [5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30]


def get_daily_interpolated(times, values, duration, step_size=1440):
    """Wandelt unregelmäßige Daten in tägliche Stützstellen um."""
    day_times = np.arange(0, duration + 1, step_size)
    interp_values = np.interp(day_times, times, values)
    return day_times, interp_values


def plot_mean_std(ax, data_list, label, color, x_axis):
    """Zeichnet Mittelwert und Standardabweichung (Volatilität)."""
    arr = np.array(data_list)
    limit = min(len(x_axis), arr.shape[1])
    arr = arr[:, :limit]
    x = x_axis[:limit]

    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)

    ax.plot(x, mean, label=label, color=color, linewidth=2)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2, label='_nolegend_')


def plot_ci_point_range(ax, data, title, unit, color):
    """Zeichnet Point & Range (95% CI)."""
    n = len(data)
    mean = np.mean(data)
    std = np.std(data)
    se = std / np.sqrt(n)
    ci = 1.96 * se

    low = mean - ci
    high = mean + ci

    ax.errorbar(x=0, y=mean, yerr=ci, fmt='o', color=color,
                ecolor=color, elinewidth=3, capsize=10,
                markersize=12, markeredgecolor='black', markeredgewidth=1.5,
                zorder=3)

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel(unit)
    ax.set_xticks([])
    ax.set_xlim(-0.5, 0.5)
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

    span = (high - low)
    buffer = span * 0.5 if span > 0 else abs(mean) * 0.1
    if buffer == 0: buffer = 1.0
    ax.set_ylim(low - buffer, high + buffer)

    if abs(mean) > 1000:
        txt_mean = f"{mean:,.0f}"
        txt_ci = f"±{ci:,.0f}"
    else:
        txt_mean = f"{mean:.2f}"
        txt_ci = f"±{ci:.2f}"

    info_text = f"Ø {txt_mean}\n({txt_ci})"
    ax.text(0.1, mean, info_text, ha='left', va='center',
            fontsize=10, fontweight='bold', color='#333333',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

    ax.text(0, low - (buffer * 0.4), f"n={n}", ha='center', va='bottom', color='gray', fontsize=9)


def perform_significance_test(results, metric_key, name_a, name_b):
    """
    Führt einen Welch's T-Test zwischen zwei Agenten durch.
    """
    if name_a not in results or name_b not in results:
        print(f"Skipping T-Test: {name_a} or {name_b} missing.")
        return

    data_a = results[name_a][metric_key]
    data_b = results[name_b][metric_key]

    # Welch's t-test (equal_var=False ist sicherer)
    t_stat, p_val = stats.ttest_ind(data_a, data_b, equal_var=False)

    mean_a = np.mean(data_a)
    mean_b = np.mean(data_b)
    diff_pct = ((mean_a - mean_b) / mean_b) * 100

    print(f"\n--- T-Test: {metric_key} ---")
    print(f"  {name_a}: {mean_a:,.1f} vs {name_b}: {mean_b:,.1f}")
    print(f"  Diff: {diff_pct:+.2f}%")
    print(f"  p-value: {p_val:.5f}")

    if p_val < 0.05:
        print("  >> ERGEBNIS: Signifikanter Unterschied! (p < 0.05)")
    else:
        print("  >> ERGEBNIS: Kein signifikanter Unterschied (Zufall möglich).")

def perform_variance_test(results, metric_key, name_a, name_b):
    """
    Führt einen Levene-Test durch, um zu prüfen, ob sich die Varianzen
    (die Volatilität/Schwankung) zweier Agenten signifikant unterscheiden.
    """
    if name_a not in results or name_b not in results:
        return

    data_a = results[name_a][metric_key]
    data_b = results[name_b][metric_key]

    # Levene-Test auf Varianzgleichheit
    stat, p_val = stats.levene(data_a, data_b)

    # Standardabweichung zur Veranschaulichung berechnen
    std_a = np.std(data_a)
    std_b = np.std(data_b)

    diff_pct = ((std_a - std_b) / std_b) * 100 if std_b > 0 else 0

    print(f"\n--- Levene-Test (Varianz/Stabilität): {metric_key} ---")
    print(f"  {name_a} StdDev: {std_a:,.1f} vs {name_b} StdDev: {std_b:,.1f}")
    print(f"  Unterschied in der Streuung: {diff_pct:+.2f}%")

    if p_val < 0.05:
        better_name = name_a if std_a < std_b else name_b
        print(f"  >> ERGEBNIS: Signifikanter Unterschied! (p = {p_val:.5f})")
        print(f"  >> FAZIT: '{better_name}' steuert signifikant stabiler (weniger Schwankungen).")
    else:
        print(f"  >> ERGEBNIS: Kein signifikanter Unterschied in der Varianz (p = {p_val:.5f}).")

def run_evaluation(policy=None, normalizer=None, n_episodes=10, name="Agent", mode="agent", base_seed=1):
    print(f"\n>>> Starte Evaluation für: {name} (Seed Base: {base_seed})")

    plan = Plan()
    plan.duration = SIM_DURATION

    # Implicit Mode Handling
    current_implicit_mode = EVAL_IMPLICIT_MODE
    if mode in ["ga", "two"]:  # Beide Heuristiken nutzen hier denselben Modus wie trainiert
        current_implicit_mode = True

    # Sicherstellen, dass Wrapper korrekt initiiert wird
    plan.use_implicit_batch = current_implicit_mode
    init_batch = {"time": 0, "number": 10}
    env = SimpyShopWrapper(plan, init_batch, use_implicit_batch=current_implicit_mode)

    # Speicher
    metrics = {
        'profits': [], 'throughputs': [], 'wip_avgs': [],
        'raw_avgs': [], 'done_avgs': [],
        'daily_wips': [], 'daily_throughputs': [], 'daily_cum_profits': [], 'on_time_rates': [], 'late_deviations': [],
        'avg_dis': [], 'avg_cln': [], 'avg_ins': [], 'avg_rep': [], 'avg_fin': [],
        'ts_dis_all': [], 'ts_cln_all': [], 'ts_ins_all': [], 'ts_rep_all': [], 'ts_fin_all': []
    }

    # Statische Action vorbereiten
    static_action = None
    if mode == "ga":
        static_action = genome_to_action(BEST_GA_GENES, GA_BATCH_SIZES)
        print(f"  [GA] Static Policy: Workers={20 + static_action.capacity_level}, Batch={static_action.batch_size}")
    elif mode == "two":
        static_action = genome_to_action(BEST_TWO_GENES, GA_BATCH_SIZES)
        print(f"  [TWO] Static Policy: Workers={20 + static_action.capacity_level}, Batch={static_action.batch_size}")

    for ep in range(n_episodes):
        raw_state = env.reset()

        # Seed
        current_seed = base_seed + ep
        env.shop.rand_mgr = RandomManager(base_seed=current_seed, config_id=0, run_id=0)

        # Norm
        if mode == "agent" and policy and normalizer:
            state = normalizer.normalize(raw_state)
        else:
            state = raw_state

        done = False
        ep_profit = 0

        ts_time = [0]
        ts_wip = [0]
        ts_tp = [0]
        ts_profit = [0]

        ts_dis, ts_cln, ts_ins, ts_rep, ts_fin = [0], [0], [0], [0], [0]

        wip_sum = 0
        raw_sum = 0
        done_sum = 0
        steps = 0

        while not done:
            # --- ACTION SELECTION ---
            if mode == "agent" and policy:
                # RL AGENT
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    actions_indices, _ = policy(state_tensor, deterministic=True)

                config = policy.config
                bs_idx = actions_indices['batch_size'].item()
                real_bs = config.batch_sizes[bs_idx]

                strat_idx, bp_low, bp_mid, bp_high = 0, 0, 0, 0
                if getattr(config, 'use_implicit_batch', False):
                    bp_low = actions_indices['batch_prio_low'].item()
                    bp_mid = actions_indices['batch_prio_mid'].item()
                    bp_high = actions_indices['batch_prio_high'].item()
                else:
                    if 'batch_strategy' in actions_indices:
                        strat_idx = actions_indices['batch_strategy'].item()

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

            elif mode in ["ga", "two"]:
                # Heuristik (Statisch)
                action = static_action

            else:
                # RANDOM
                action = Action(
                    prio_disassembly=np.random.randint(0, 6),
                    prio_inspection=np.random.randint(0, 6),
                    prio_cleaning=np.random.randint(0, 6),
                    prio_repair=np.random.randint(0, 6),
                    prio_assembly=np.random.randint(0, 6),
                    order_release=np.random.randint(0, 5),
                    batch_size=np.random.choice([5, 10, 15, 20, 30]),
                    capacity_level=np.random.randint(0, 6),
                    batch_strategy=0,
                    batch_prio_low=np.random.randint(0, 6),
                    batch_prio_mid=np.random.randint(0, 6),
                    batch_prio_high=np.random.randint(0, 6)
                )

            # --- STEP ---
            raw_next_state, reward, done, info = env.step(action)

            if mode == "agent" and policy and normalizer:
                state = normalizer.normalize(raw_next_state)
            else:
                state = raw_next_state

            ep_profit += info.get('step_profit', 0)

            # Logging
            ts_time.append(env.env.now)
            ts_wip.append(info.get('wip_count', 0))
            ts_tp.append(env.shop.total_throughput)
            ts_profit.append(ep_profit)

            ts_dis.append(len(env.shop.disassembled.items))
            ts_cln.append(len(env.shop.cleaned.items))
            ts_ins.append(len(env.shop.inspected.items))
            ts_rep.append(len(env.shop.repaired.items))
            ts_fin.append(len(env.shop.finished.items))

            wip_sum += info.get('wip_count', 0)
            curr_raw = len(env.shop.buffer_q0.items) + len(env.shop.buffer_q1.items) + len(env.shop.buffer_q2.items)
            curr_done = len(env.shop.product_done.items)
            raw_sum += curr_raw
            done_sum += curr_done

            steps += 1

            if done:
                # Wir greifen auf die Kunden/Auftrags-Daten im ShopFloor zu.
                # (Passe den Variablennamen an, falls er in internal.py anders heißt)
                if hasattr(env.shop, 'tardiness_history') and len(env.shop.tardiness_history) > 0:
                    tardiness = np.array(env.shop.tardiness_history)

                    # 1. Liefertreue in % (Tardiness <= 0 bedeutet pünktlich oder zu früh)
                    on_time_pct = np.mean(tardiness <= 0) * 100.0

                    # 2. Durchschnittliche Abweichung NUR bei verspäteten Aufträgen
                    late_only = tardiness[tardiness > 0]

                    # Angenommen, Tardiness ist in Minuten -> Umrechnung in Tage
                    if len(late_only) > 0:
                        mean_late_days = np.mean(late_only) / (24 * 60)
                    else:
                        mean_late_days = 0.0

                else:
                    # Fallback, falls die Historie (noch) nicht existiert
                    on_time_pct = 0.0
                    mean_late_days = 0.0

                metrics['on_time_rates'].append(on_time_pct)
                metrics['late_deviations'].append(mean_late_days)

        # --- POST EPISODE ---
        metrics['profits'].append(ep_profit)
        metrics['throughputs'].append(info['throughput'])
        metrics['wip_avgs'].append(wip_sum / steps if steps else 0)
        metrics['raw_avgs'].append(raw_sum / steps if steps else 0)
        metrics['done_avgs'].append(done_sum / steps if steps else 0)

        _, daily_wip = get_daily_interpolated(ts_time, ts_wip, SIM_DURATION)
        _, daily_tp_cum = get_daily_interpolated(ts_time, ts_tp, SIM_DURATION)
        _, daily_prof_cum = get_daily_interpolated(ts_time, ts_profit, SIM_DURATION)
        daily_tp_delta = np.diff(daily_tp_cum, prepend=0)

        metrics['daily_wips'].append(daily_wip)
        metrics['daily_throughputs'].append(daily_tp_delta)
        metrics['daily_cum_profits'].append(daily_prof_cum)

        # Puffer-Durchschnitte
        metrics['avg_dis'].append(np.mean(ts_dis))
        metrics['avg_cln'].append(np.mean(ts_cln))
        metrics['avg_ins'].append(np.mean(ts_ins))
        metrics['avg_rep'].append(np.mean(ts_rep))
        metrics['avg_fin'].append(np.mean(ts_fin))

        _, daily_dis = get_daily_interpolated(ts_time, ts_dis, SIM_DURATION)
        _, daily_cln = get_daily_interpolated(ts_time, ts_cln, SIM_DURATION)
        _, daily_ins = get_daily_interpolated(ts_time, ts_ins, SIM_DURATION)
        _, daily_rep = get_daily_interpolated(ts_time, ts_rep, SIM_DURATION)
        _, daily_fin = get_daily_interpolated(ts_time, ts_fin, SIM_DURATION)

        metrics['ts_dis_all'].append(daily_dis)
        metrics['ts_cln_all'].append(daily_cln)
        metrics['ts_ins_all'].append(daily_ins)
        metrics['ts_rep_all'].append(daily_rep)
        metrics['ts_fin_all'].append(daily_fin)

        if (ep + 1) % 5 == 0:
            print(f"  Ep {ep + 1}/{n_episodes}: Profit={ep_profit / 1000:.1f}k")

    return metrics


def main():
    # --- SETUP ---
    dummy_plan = Plan()
    dummy_env = SimpyShopWrapper(dummy_plan, {"time": 0, "number": 10}, use_implicit_batch=EVAL_IMPLICIT_MODE)
    dummy_state = dummy_env.reset()
    detected_dim = len(dummy_state)
    del dummy_env

    config = PPOConfig(input_dim=detected_dim, use_implicit_batch=EVAL_IMPLICIT_MODE)

    model_path = os.path.join("models", "policy_step_212.pt")
    norm_path = os.path.join("models", "normalizer_step_212.npz")

    policy = None
    normalizer = None

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

    n_test = 20
    BASE_SEED = 1000
    results = {}

    # 1. PPO Agent
    if policy:
        results['PPO Agent'] = run_evaluation(policy, normalizer, n_test, "PPO Agent", "agent", base_seed=BASE_SEED)

    # 2. GA Heuristik
    if COMPARE_WITH_GA:
        results['GA (Evolution)'] = run_evaluation(None, None, n_test, "GA Heuristik", "ga", base_seed=BASE_SEED)

    # 3. TWO Heuristik
    if COMPARE_WITH_TWO:
        results['TWO (Physics)'] = run_evaluation(None, None, n_test, "TWO Heuristik", "two", base_seed=BASE_SEED)

    # 4. Random
    if COMPARE_WITH_RANDOM:
        results['Random'] = run_evaluation(None, None, n_test, "Random", "random", base_seed=BASE_SEED)

    # --- PLOTTING ---
    colors = {'PPO Agent': 'tab:green', 'GA (Evolution)': 'tab:blue', 'TWO (Physics)': 'tab:red', 'Random': 'tab:gray'}
    names = list(results.keys())

    # ===========================================================
    # PLOT 1: Vergleich (Boxplots)
    # ===========================================================
    print("\nErstelle Vergleichs-Plot...")
    fig1, axs = plt.subplots(1, 5, figsize=(20, 6))
    fig1.suptitle(f'Benchmark Vergleich (n={n_test} Episoden)', fontsize=16)

    def create_bp_scatter(ax, key, title, ylabel):
        data = [results[n][key] for n in names]
        bp = ax.boxplot(data, labels=names, patch_artist=True, showfliers=False)
        for patch, name in zip(bp['boxes'], names):
            patch.set_facecolor(colors[name])
            patch.set_alpha(0.6)

        for i, d in enumerate(data):
            y = d
            x = np.random.normal(i + 1, 0.04, size=len(y))
            ax.scatter(x, y, color='black', alpha=0.5, s=20, zorder=3)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    create_bp_scatter(axs[0], 'profits', 'Gesamtprofit', '€')
    create_bp_scatter(axs[1], 'throughputs', 'Gesamtdurchsatz', 'Stück')
    create_bp_scatter(axs[2], 'wip_avgs', 'Ø WIP Bestand', 'Stück')
    create_bp_scatter(axs[3], 'raw_avgs', 'Ø Rohmateriallager', 'Stück')
    create_bp_scatter(axs[4], 'done_avgs', 'Ø Fertigwarenlager', 'Stück')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("benchmark_comparison.png")
    print("-> gespeichert: benchmark_comparison.png")

    # ===========================================================
    # PLOT 2: Verlauf
    # ===========================================================
    print("\nErstelle Verlauf-Analyse...")
    days_x = np.arange(0, DAYS_TO_PLOT + 1)

    fig2, axs2 = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    fig2.suptitle(f'Verlauf & Volatilität (Mean ± Std)', fontsize=16)

    def plot_metric_for_all(ax, metric_key, title, ylabel):
        for name in names:
            data_list = results[name][metric_key]
            plot_mean_std(ax, data_list, name, colors[name], days_x)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')

    plot_metric_for_all(axs2[0], 'daily_wips', 'Täglicher WIP Verlauf', 'Anzahl Teile')
    plot_metric_for_all(axs2[1], 'daily_throughputs', 'Täglicher Output', 'Stück / Tag')
    plot_metric_for_all(axs2[2], 'daily_cum_profits', 'Kumulierter Profit', '€')

    axs2[2].axhline(0, color='black', linewidth=1, linestyle='--')
    axs2[2].set_xlabel("Tag der Simulation")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("benchmark_volatility.png")
    print("-> gespeichert: benchmark_volatility.png")

    # ===========================================================
    # PLOT 3: CI
    # ===========================================================
    print("\nErstelle Statistik (CI)...")
    fig3, axs3 = plt.subplots(1, 3, figsize=(16, 6))
    fig3.suptitle(f'Statistische Sicherheit (Mean ± 95% CI)', fontsize=16)

    def plot_multi_ci(ax, key, title, unit):
        for i, name in enumerate(names):
            data = results[name][key]
            n = len(data)
            mean = np.mean(data)
            std = np.std(data)
            se = std / np.sqrt(n)
            ci = 1.96 * se

            col = colors[name]
            ax.errorbar(x=i, y=mean, yerr=ci, fmt='o', color=col,
                        ecolor=col, elinewidth=3, capsize=10,
                        markersize=12, markeredgecolor='black', markeredgewidth=1.5,
                        zorder=3)

            if abs(mean) > 1000:
                txt = f"{mean:,.0f}"
            else:
                txt = f"{mean:.1f}"
            ax.text(i + 0.1, mean, txt, ha='left', va='center', fontweight='bold', fontsize=10)

        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        all_data = []
        for n in names: all_data.extend(results[n][key])
        mi, ma = min(all_data), max(all_data)
        margin = (ma - mi) * 0.2
        if margin == 0: margin = 10
        ax.set_ylim(mi - margin, ma + margin)

    plot_multi_ci(axs3[0], 'profits', 'Gesamtprofit', '€')
    plot_multi_ci(axs3[1], 'throughputs', 'Gesamtdurchsatz', 'Stück')
    plot_multi_ci(axs3[2], 'wip_avgs', 'Ø WIP Bestand', 'Teile')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig("benchmark_ci.png")
    print("-> gespeichert: benchmark_ci.png")

    # ===========================================================
    # PLOT 4: Liefertreue (Delivery Reliability)
    # ===========================================================
    print("\nErstelle Liefertreue-Plot...")
    fig4, axs4 = plt.subplots(1, 2, figsize=(14, 6))
    fig4.suptitle(f'Kundenperspektive: Liefertreue & Verspätungen (n={n_test} Episoden)', fontsize=16)

    # Subplot 1: Liefertreue in %
    create_bp_scatter(axs4[0], 'on_time_rates', 'Liefertreue (On-Time Delivery)', 'Prozent (%)')
    axs4[0].set_ylim(-5, 105)  # Prozente gehen von 0 bis 100
    # Horizontale Ziel-Linie einzeichnen (z.B. 95% Service Level)
    axs4[0].axhline(95, color='green', linestyle='--', alpha=0.5, label='Ziel: 95%')
    axs4[0].legend(loc='lower right')

    # Subplot 2: Absolute Abweichung (in Tagen) bei Terminüberschreitung
    create_bp_scatter(axs4[1], 'late_deviations', 'Ø Abweichung (nur bei Verspätung)', 'Tage zu spät')
    axs4[1].set_ylim(bottom=0)  # Keine negativen Werte zulassen
    axs4[1].axhline(2.0, color='red', linestyle='--', alpha=0.5, label='Kritisch (> 2 Tage)')
    axs4[1].legend(loc='upper right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("benchmark_delivery.png")
    print("-> gespeichert: benchmark_delivery.png")

    # ===========================================================
    # PLOT 5: Puffer-Bestände & Line Balance (Bottleneck Analyse)
    # ===========================================================
    print("\nErstelle Puffer-Bestand-Plot (Balance-Check)...")
    fig5 = plt.figure(figsize=(20, 10))
    fig5.suptitle(f'WIP-Verteilung & Bottleneck Analyse (n={n_test} Episoden)', fontsize=16)

    # --- Oberer Bereich: Balkendiagramm der Durchschnitte ---
    ax_avg = plt.subplot(2, 1, 1)

    stages = ['Disassembled', 'Cleaned', 'Inspected', 'Repaired', 'Finished (Sub-Comps)']
    n_agents = len(names)
    bar_width = 0.8 / n_agents
    x_pos = np.arange(len(stages))

    for i, name in enumerate(names):
        avg_vals = [
            np.mean(results[name]['avg_dis']),
            np.mean(results[name]['avg_cln']),
            np.mean(results[name]['avg_ins']),
            np.mean(results[name]['avg_rep']),
            np.mean(results[name]['avg_fin'])
        ]
        # Versetzte Balken für jeden Agenten nebeneinander zeichnen
        ax_avg.bar(x_pos + i * bar_width - (0.8 / 2) + (bar_width / 2), avg_vals, bar_width,
                   label=name, color=colors[name], alpha=0.8, edgecolor='black')

    ax_avg.set_title("Durchschnittlicher Bestand pro Puffer-Station (Wo entsteht der Stau?)", fontweight='bold')
    ax_avg.set_ylabel("Ø Anzahl Teile im Puffer")
    ax_avg.set_xticks(x_pos)
    ax_avg.set_xticklabels(stages, fontsize=11)
    ax_avg.legend(loc='upper right')
    ax_avg.grid(axis='y', linestyle='--', alpha=0.5)

    # --- Unterer Bereich: Verlauf pro Station (5 Subplots) ---
    # Wir teilen die untere Hälfte in 5 Spalten (2. Zeile, 5 Spalten, Index 6 bis 10)
    ax_ts_dis = plt.subplot(2, 5, 6)
    ax_ts_cln = plt.subplot(2, 5, 7, sharey=ax_ts_dis)
    ax_ts_ins = plt.subplot(2, 5, 8, sharey=ax_ts_dis)
    ax_ts_rep = plt.subplot(2, 5, 9, sharey=ax_ts_dis)
    ax_ts_fin = plt.subplot(2, 5, 10, sharey=ax_ts_dis)

    ts_axes = [ax_ts_dis, ax_ts_cln, ax_ts_ins, ax_ts_rep, ax_ts_fin]
    ts_keys = ['ts_dis_all', 'ts_cln_all', 'ts_ins_all', 'ts_rep_all', 'ts_fin_all']
    stage_names = ['Demontiert', 'Gereinigt', 'Geprüft', 'Repariert', 'Fertige Komponenten']

    days_x = np.arange(0, DAYS_TO_PLOT + 1)

    for ax, key, title in zip(ts_axes, ts_keys, stage_names):
        for name in names:
            plot_mean_std(ax, results[name][key], name, colors[name], days_x)
        ax.set_title(f"Verlauf: {title}", fontsize=10)
        ax.set_xlabel("Tag")
        ax.grid(True, alpha=0.3)
        # Verstecke Y-Achsen Labels für alle außer dem ganz linken Plot für mehr Sauberkeit
        if ax != ax_ts_dis:
            plt.setp(ax.get_yticklabels(), visible=False)

    ax_ts_dis.set_ylabel("Teile im Puffer")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("benchmark_buffers.png")
    print("-> gespeichert: benchmark_buffers.png")

    # ===========================================================
    # STATISTISCHE TESTS (VOR PLT.SHOW!)
    # ===========================================================
    print("\n" + "=" * 40)
    print("STATISTISCHE ANALYSE")
    print("=" * 40)

    # 1. PPO vs GA
    if 'PPO Agent' in results and 'GA (Evolution)' in results:
        print("\n--- PPO vs GA ---")
        perform_significance_test(results, 'profits', 'PPO Agent', 'GA (Evolution)')
        perform_significance_test(results, 'throughputs', 'PPO Agent', 'GA (Evolution)')
        perform_significance_test(results, 'wip_avgs', 'PPO Agent', 'GA (Evolution)')

    # 2. PPO vs TWO (wenn vorhanden)
    if 'PPO Agent' in results and 'TWO (Physics)' in results:
        print("\n--- PPO vs TWO ---")
        perform_significance_test(results, 'profits', 'PPO Agent', 'TWO (Physics)')
        perform_significance_test(results, 'throughputs', 'PPO Agent', 'TWO (Physics)')
        perform_significance_test(results, 'wip_avgs', 'PPO Agent', 'TWO (Physics)')

    # 3. GA vs TWO (Heuristik Battle)
    if 'GA (Evolution)' in results and 'TWO (Physics)' in results:
        print("\n--- GA vs TWO ---")
        perform_significance_test(results, 'profits', 'GA (Evolution)', 'TWO (Physics)')
        perform_significance_test(results, 'throughputs', 'GA (Evolution)', 'TWO (Physics)')
        perform_significance_test(results, 'wip_avgs', 'GA (Evolution)', 'TWO (Physics)')

    print("\n" + "=" * 40)
    print("STATISTISCHE ANALYSE (VARIANZ / STABILITÄT)")
    print("=" * 40)

    # Varianz-Test: PPO vs GA
    if 'PPO Agent' in results and 'GA (Evolution)' in results:
        perform_variance_test(results, 'wip_avgs', 'PPO Agent', 'GA (Evolution)')

    # Varianz-Test: PPO vs TWO
    if 'PPO Agent' in results and 'TWO (Physics)' in results:
        perform_variance_test(results, 'wip_avgs', 'PPO Agent', 'TWO (Physics)')

    # Varianz-Test: GA vs TWO
    if 'GA (Evolution)' in results and 'TWO (Physics)' in results:
        perform_variance_test(results, 'wip_avgs', 'GA (Evolution)', 'TWO (Physics)')

    plt.show()


if __name__ == "__main__":
    main()