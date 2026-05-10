# learning/shield.py
import numpy as np
from dataclasses import dataclass
from learning.action import Action


@dataclass
class ShieldCorrection:
    safe_action: Action
    penalty: float
    was_corrected: bool


class ProductionShield:
    def __init__(self, config):
        self.config = config  # SimPy Plan Objekt
        self.default_workers = 20
        self.penalty_factor = 0.2

    def protect(self, raw_action: Action, current_labor_budget: int = 20, shop_state=None,
                planned_allocation=None) -> ShieldCorrection:
        penalty = 0.0
        was_corrected = False

        # --- 1. SETUP & CAPS ---
        # Stationen mappen (Index -> Name -> Cap)
        # Reihenfolge: 0:Insp, 1:Dis, 2:Clean, 3:Rep, 4:Ass
        station_caps = [
            self.config.inspection_bench + self.config.test_bench_inspec,
            self.config.workstation_dis,
            self.config.washing_machine + self.config.sand_blaster,
            self.config.machines_rep,
            self.config.assembly_station + self.config.painting_booth
        ]

        # Wir arbeiten mit einer Liste für die Prios
        mod_prios = [
            int(raw_action.prio_inspection),
            int(raw_action.prio_disassembly),
            int(raw_action.prio_cleaning),
            int(raw_action.prio_repair),
            int(raw_action.prio_assembly)
        ]

        # Order Release separat speichern, da wir es ggf. ändern
        safe_order_release = int(raw_action.order_release)

        # --- 2. LOGIK-ANPASSUNGEN (VOR der Verteilung!) ---

        # A. Deadlock Prevention (Falls alles 0 ist)
        if sum(mod_prios) == 0:
            mod_prios = [2, 2, 2, 2, 2]
            penalty += 1.0
            was_corrected = True

        # B. Assembly Logic Fix (State Aware)
        # Wir ändern die Priorität BEVOR wir verteilen, damit die Kaskade das berücksichtigt.
        if shop_state:
            def count_in_store(store, t_char):
                return sum(1 for item in store.items if getattr(item, 'type', '') == t_char)

            # Check: Liegt Material für Montage bereit? (Repaired + Finished Puffer)
            has_s = count_in_store(shop_state.repaired, 'S') + count_in_store(shop_state.finished, 'S')
            has_r = count_in_store(shop_state.repaired, 'R') + count_in_store(shop_state.finished, 'R')
            has_p = count_in_store(shop_state.repaired, 'P') + count_in_store(shop_state.finished, 'P')
            has_m = count_in_store(shop_state.repaired, 'M') + count_in_store(shop_state.finished, 'M')
            has_c = count_in_store(shop_state.repaired, 'C') + count_in_store(shop_state.finished, 'C')
            has_e = count_in_store(shop_state.repaired, 'E') + count_in_store(shop_state.finished, 'E')

            min_parts = min(has_s, has_r, has_p, has_m, has_c, has_e)

            # Fall 1: Montage möglich, aber Prio niedrig -> Force High
            if min_parts > 0 and mod_prios[4] < 2:
                mod_prios[4] = 5  # Assembly Prio hochsetzen
                penalty += 0.2
                was_corrected = True

            # Fall 2: Keine Sets da, aber Prio hoch -> Drosseln (spart Worker)
            if min_parts == 0 and mod_prios[4] > 1:
                mod_prios[4] = 1  # Minimum Keep-Alive
                was_corrected = True

        # C. WIP-Cap für Order Release
        if shop_state:
            wip = (len(shop_state.disassembled.items) +
                   len(shop_state.cleaned.items) +
                   len(shop_state.repaired.items))

            if wip > 120 and safe_order_release > 0:
                safe_order_release = 0
                penalty += 0.2
                was_corrected = True

        # --- 3. CASCADING REDISTRIBUTION (Hard Clipping) ---
        # Jetzt verteilen wir basierend auf den MODIFIZIERTEN Prios (mod_prios)

        final_counts = [0] * 5
        workers_left = current_labor_budget

        # Sicherheits-Loop
        for _ in range(10):
            if workers_left <= 0: break

            total_prio = sum(mod_prios)
            if total_prio == 0: break

            distributed_this_round = [0] * 5
            for i in range(5):
                if mod_prios[i] > 0:
                    # Anteil berechnen
                    share = int(workers_left * (mod_prios[i] / total_prio))
                    if share == 0 and workers_left > 0 and mod_prios[i] > 0:
                        share = 1

                    # Check gegen Cap
                    space_left = station_caps[i] - final_counts[i]
                    can_take = max(0, min(share, space_left))

                    final_counts[i] += can_take
                    distributed_this_round[i] = can_take

            total_dist = sum(distributed_this_round)
            workers_left -= total_dist

            # Volle Stationen für nächste Runde deaktivieren
            for i in range(5):
                if final_counts[i] >= station_caps[i]:
                    mod_prios[i] = 0

            if total_dist == 0:
                break

        # --- 4. OVERSTAFFING PENALTY ---
        # Prüfen, ob der Agent ursprünglich mehr wollte als physikalisch möglich
        # (Lerneffekt, damit er nicht blind max_prio spammt)
        orig_wants = [
            int(raw_action.prio_inspection),
            int(raw_action.prio_disassembly),
            int(raw_action.prio_cleaning),
            int(raw_action.prio_repair),
            int(raw_action.prio_assembly)
        ]

        for i in range(5):
            # Wenn Agent mehr wollte als am Ende zugewiesen wurde UND die Station voll ist
            if orig_wants[i] > final_counts[i] and final_counts[i] == station_caps[i]:
                # Kleine Strafe für Verschwendung
                penalty += 0.05

         # --- 5. RESULT ---
        safe_action = Action(
            prio_inspection=final_counts[0],
            prio_disassembly=final_counts[1],
            prio_cleaning=final_counts[2],
            prio_repair=final_counts[3],
            prio_assembly=final_counts[4],

            order_release=safe_order_release,

            batch_size=max(2, int(raw_action.batch_size)),
            capacity_level=max(0, int(raw_action.capacity_level)),
            batch_strategy=int(raw_action.batch_strategy),
            batch_prio_low=raw_action.batch_prio_low,
            batch_prio_mid=raw_action.batch_prio_mid,
            batch_prio_high=raw_action.batch_prio_high
        )

        return ShieldCorrection(safe_action, penalty, was_corrected)