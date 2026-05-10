# learning/simpy_shop_wrapper.py
import numpy as np
import simpy
from simulation.internal import ShopFloor
from learning.action import Action
from learning.shield import ProductionShield
from learning.context_extractor import ContextExtractor, HistoricalRecord
from simulation.random_manager import RandomManager

class SimpyShopWrapper(object):
    """
    Verbindet die SimPy-Simulation mit dem RL-Agenten.
    Inklusive Continuous Accounting, Shielding, Context-Extraction und Micro-Trends.
    """

    def __init__(self, plan, init_batches, use_implicit_batch=False, base_seed=100):
        self.plan = plan
        self.init_batches = init_batches

        # Seed Management für RL Training
        self.base_seed = base_seed
        self.episode_counter = 0

        # SimPy Objekte
        self.env = None
        self.shop = None

        # Shield initialisieren
        self.shield = ProductionShield(config=self.plan)

        # Context Extractor
        self.context_extractor = ContextExtractor(window_days=30, aggregation_hours=24)

        # Reward Tracking
        self._last_cost = 0
        self._last_rev = 0
        self.total_reward = 0

        self.state_dim = 50  # wird dynamisch überschrieben

        # Tracking für Context Updates (Makro)
        self._prev_low = 0
        self._prev_high = 0
        self._prev_done_len = 0
        self._prev_discard_len = 0

        # --- NEU: Tracking für Micro-Trends (Deltas) ---
        self._prev_backlog_micro = 0
        self._prev_wip_micro = 0
        self._prev_raw_micro = 0
        # -----------------------------------------------

        # Batchstrategie Modus
        self.use_implicit_batch = use_implicit_batch

    def _bottleneck_watchdog(self):
        """
        Ein Hintergrund-Prozess, der Puffer überwacht und den Agenten ruft,
        wenn sich Material staut (WIP > 30).
        """
        monitor_list = {
            "core_queue": self.shop.core_queue,
            "disassembled": self.shop.disassembled,
            "cleaned": self.shop.cleaned,
            "inspected": self.shop.inspected,
            "repaired": self.shop.repaired,
            "finished": self.shop.finished
        }

        active_alarms = {name: False for name in monitor_list}
        HIGH_THRESHOLD = 30
        LOW_THRESHOLD = 24
        CHECK_INTERVAL = 240

        while True:
            yield self.env.timeout(CHECK_INTERVAL)
            should_trigger_agent = False

            for name, store in monitor_list.items():
                current_wip = len(store.items)
                is_active = active_alarms[name]

                if not is_active:
                    if current_wip > HIGH_THRESHOLD:
                        active_alarms[name] = True
                        should_trigger_agent = True
                else:
                    if current_wip <= LOW_THRESHOLD:
                        active_alarms[name] = False

            if should_trigger_agent:
                self.shop.ctrl.pause_2()

    def reset(self):
        self.env = simpy.Environment()
        self.episode_counter += 1
        current_rand_mgr = RandomManager(base_seed=self.base_seed, run_id=self.episode_counter)

        self.shop = ShopFloor(self.env, self.plan, logging_enabled=False)
        self.shop.rand_mgr = current_rand_mgr
        self.shop.batches = self.init_batches.copy()

        self.shop.batch_mix_stats = []

        # Reset Context Tracking
        self._prev_low = 0
        self._prev_high = 0
        self._prev_done_len = 0
        self._prev_discard_len = 0

        # --- NEU: Reset Micro-Trend Tracking ---
        self._prev_backlog_micro = 0
        self._prev_wip_micro = 0
        self._prev_raw_micro = 0
        # ---------------------------------------

        self.context_extractor = ContextExtractor(window_days=30, aggregation_hours=24)

        self.env.process(self._bottleneck_watchdog())

        c, r, _ = self.shop.get_snapshot()
        self._last_cost = c
        self._last_rev = r
        self.total_reward = 0

        return self.get_state_vector()

    def get_machine_cost_per_minute(self):
        cost_map_1500 = {
            'workstation_dis': 10000,
            'washing_machine': 3500,
            'sand_blaster': 2000,
            'drier': 3000,
            'test_bench_inspec': 5000,
            'inspection_bench': 1000,
            'metrology': 600,
            'machines_rep': 4000,
            'painting_booth': 8000,
            'assembly_station': 1000
        }

        total_period_cost = 0
        for attr, cost in cost_map_1500.items():
            count = getattr(self.plan, attr, 0)
            total_period_cost += count * cost

        return total_period_cost / 1500.0

    def step(self, raw_action: Action):
        # --- 1. Kapazität skalieren ---
        cap_add = max(0, min(int(raw_action.capacity_level), 5))
        new_total_workers = 20 + cap_add
        self.shop.workers.set_capacity(new_total_workers)

        # --- 2. Allocation Vorbereitung ---
        prios = [
            raw_action.prio_inspection,
            raw_action.prio_disassembly,
            raw_action.prio_cleaning,
            raw_action.prio_repair,
            raw_action.prio_assembly
        ]
        total_prio = sum(prios) if sum(prios) > 0 else 1
        estimated_counts = [int((p / total_prio) * new_total_workers) for p in prios]

        # --- 3. Shielding ---
        correction = self.shield.protect(
            raw_action,
            current_labor_budget=new_total_workers,
            shop_state=self.shop,
            planned_allocation=estimated_counts
        )
        action = correction.safe_action
        shield_penalty = correction.penalty

        # --- 4. Allocation (Finale Verteilung) ---
        final_prios = [
            action.prio_inspection,
            action.prio_disassembly,
            action.prio_cleaning,
            action.prio_repair,
            action.prio_assembly
        ]
        total_prio_final = sum(final_prios)
        if total_prio_final == 0: total_prio_final = 1

        counts = [int((p / total_prio_final) * new_total_workers) for p in final_prios]
        remainder = new_total_workers - sum(counts)
        prio_indices = np.argsort(final_prios)[::-1]

        # Maschinen-Limits definieren
        station_caps = [
            self.plan.inspection_bench + self.plan.test_bench_inspec,
            self.plan.workstation_dis,
            self.plan.washing_machine + self.plan.sand_blaster,
            self.plan.machines_rep,
            self.plan.assembly_station + self.plan.painting_booth
        ]

        # Sichere Rest-Verteilung
        assigned = 0
        loop_counter = 0
        while assigned < remainder:
            target_idx = prio_indices[loop_counter % 5]

            # Prüfen, ob an dieser Station noch eine Maschine frei ist
            if counts[target_idx] < station_caps[target_idx]:
                counts[target_idx] += 1
                assigned += 1

            loop_counter += 1

            # Sicherheitsabbruch
            if loop_counter > 5:
                counts[4] += (remainder - assigned)
                break

        self.shop.workers.reallocate(counts)

        # --- 5. Supply & Batch Parameter ---
        self.shop.order_release_from_action = int(action.order_release)
        self.shop.batch_size_from_action = int(action.batch_size)

        if self.use_implicit_batch:
            self.shop.batch_strategy_from_action = 99
            raw_weights = np.array([
                float(action.batch_prio_low),
                float(action.batch_prio_mid),
                float(action.batch_prio_high)
            ], dtype=np.float32)

            total = np.sum(raw_weights)
            if total > 0:
                probs = raw_weights / total
            else:
                probs = np.array([0.33, 0.33, 0.33])

            if isinstance(probs, np.ndarray):
                self.shop.batch_weights_from_action = probs.tolist()
            else:
                self.shop.batch_weights_from_action = list(probs)
        else:
            if hasattr(action, 'batch_strategy'):
                self.shop.batch_strategy_from_action = int(action.batch_strategy)
            else:
                self.shop.batch_strategy_from_action = 0

        # --- SNAPSHOT VOR DEM RUN ---
        c_prev, r_prev, _ = self.shop.get_snapshot()

        # --- 6. Simulation Step ---
        start_time = self.env.now
        try:
            triggers = [self.shop.ctrl.event, self.shop.ctrl.event_2]
            self.env.run(until=simpy.AnyOf(self.env, triggers))
        except simpy.Interrupt:
            pass

        duration = self.env.now - start_time

        # --- 7. Context Update ---
        curr_low = sum(1 for i in self.shop.supplies_t.items if i.quality in [0, 1])
        curr_high = sum(1 for i in self.shop.supplies_t.items if i.quality == 2)
        arr_low = max(0, curr_low - self._prev_low)
        arr_high = max(0, curr_high - self._prev_high)
        self._prev_low, self._prev_high = curr_low, curr_high

        curr_done = len(self.shop.product_done.items)
        orders_completed = max(0, curr_done - self._prev_done_len)
        self._prev_done_len = curr_done

        curr_discard = len(self.shop.discard.items)
        discarded_new = max(0, curr_discard - self._prev_discard_len)
        self._prev_discard_len = curr_discard

        total_processed = orders_completed + discarded_new
        quality_yield = orders_completed / total_processed if total_processed > 0 else 0.8

        record = HistoricalRecord(
            timestamp=self.env.now / (24 * 60),
            cores_arrived={'high': arr_high, 'low': arr_low},
            orders_completed=orders_completed,
            quality_yield=quality_yield,
            utilization=1.0,
            backlog=len(self.shop.customers)
        )
        self.context_extractor.update_history(record)

        # --- 8. CONTINUOUS ACCOUNTING ---
        c_now, r_now, _ = self.shop.get_snapshot()
        variable_cost_delta = c_now - c_prev
        revenue_delta = r_now - r_prev

        base_rate_per_min = 0.25
        overtime_factor = 1.5

        if new_total_workers <= 20:
            labor_cost_rate = new_total_workers * base_rate_per_min
        else:
            base_part = 20 * base_rate_per_min
            extra_part = (new_total_workers - 20) * (base_rate_per_min * overtime_factor)
            labor_cost_rate = base_part + extra_part

        fixed_labor_cost = labor_cost_rate * duration

        machine_cost_rate = self.get_machine_cost_per_minute()
        fixed_machine_cost = machine_cost_rate * duration

        true_profit = revenue_delta - variable_cost_delta - fixed_labor_cost - fixed_machine_cost
        controllable_profit = revenue_delta - variable_cost_delta - fixed_labor_cost

        # --- 9. REWARD CALCULATION ---
        scaled_controllable_profit = controllable_profit / 50000.0

        q_raw = len(self.shop.buffer_q0.items) + \
                len(self.shop.buffer_q1.items) + \
                len(self.shop.buffer_q2.items)
        penalty_raw = q_raw * 0.0005

        q_done = len(self.shop.product_done.items)
        penalty_done = q_done * 0.005

        count_wip_parts = (len(self.shop.disassembled.items) +
                           len(self.shop.cleaned.items) +
                           len(self.shop.inspected.items) +
                           len(self.shop.repaired.items) +
                           len(self.shop.finished.items))

        wip_in_cores = count_wip_parts / 6.0
        soft_limit = 50.0

        if wip_in_cores > soft_limit:
            excess = wip_in_cores - soft_limit
            penalty_wip = (soft_limit * 0.003) + (excess * 0.005)
        else:
            penalty_wip = wip_in_cores * 0.003

        total_inv_penalty = penalty_raw + penalty_wip + penalty_done

        throughput_bonus = (self.shop.total_throughput - self.shop.vectors[31][-1]) if len(
            self.shop.vectors[31]) > 0 else 0
        throughput_reward = throughput_bonus * 0.05

        last_mix = self.shop.batch_mix_stats[-1] if self.shop.batch_mix_stats else 0
        mix_reward = last_mix * 0.0

        reward = scaled_controllable_profit - shield_penalty - total_inv_penalty + throughput_reward + mix_reward
        reward = np.clip(reward, -10.0, 10.0)

        self._last_cost = c_now
        self._last_rev = r_now
        self.total_reward += reward

        state = self.get_state_vector()
        done = (self.env.now >= self.plan.duration)

        info = {
            'time': self.env.now,
            'duration': duration,
            'step_profit': true_profit,
            'wip_count': count_wip_parts,
            'controllable_profit': controllable_profit,
            'throughput': self.shop.total_throughput,
            'workers': new_total_workers,
            'inv_penalty': total_inv_penalty,
            'penalty_machine_overload': shield_penalty,
            'wip_core_units': wip_in_cores
        }

        return state, reward, done, info

    def get_state_vector(self):
        s = self.shop

        def get_type_counts(store):
            counts = {'S': 0, 'R': 0, 'P': 0, 'C': 0, 'E': 0, 'M': 0}
            for item in store.items:
                t = getattr(item, 'type', None)
                if t in counts:
                    counts[t] += 1
            return [counts['S'], counts['R'], counts['P'], counts['C'], counts['E'], counts['M']]

        def get_len(store):
            return len(store.items)

        # 1. Input Buffers
        q_low = get_len(s.buffer_q0)
        q_mid = get_len(s.buffer_q1)
        q_high = get_len(s.buffer_q2)
        total_supplies = q_low + q_mid + q_high
        base_queues = [total_supplies, q_low, q_mid, q_high, get_len(s.core_queue)]

        # 2. Granulare WIP Stores
        vec_dis = get_type_counts(s.disassembled)
        vec_cln = get_type_counts(s.cleaned)
        vec_ins = get_type_counts(s.inspected)
        vec_rep = get_type_counts(s.repaired)
        vec_fin_comp = get_type_counts(s.finished)

        # 3. Verfügbarkeit
        min_sets_clean = min(vec_dis)
        total_avail_ass = [r + f for r, f in zip(vec_rep, vec_fin_comp)]
        min_sets_ass = min(total_avail_ass)
        vec_sets = [min_sets_clean, min_sets_ass]

        # 4. Output & Ressourcen
        vec_out = [get_len(s.product_done)]
        workers = list(s.workers.current_allocation.values())
        machines = [
            s.plan.inspection_bench + s.plan.test_bench_inspec,
            s.plan.workstation_dis,
            s.plan.washing_machine,
            s.plan.machines_rep,
            s.plan.painting_booth
        ]

        # Alles in Arrays umwandeln
        raw_queues = np.array(base_queues, dtype=np.float32)
        raw_dis = np.array(vec_dis, dtype=np.float32)
        raw_cln = np.array(vec_cln, dtype=np.float32)
        raw_ins = np.array(vec_ins, dtype=np.float32)
        raw_rep = np.array(vec_rep, dtype=np.float32)
        raw_fin_comp = np.array(vec_fin_comp, dtype=np.float32)
        raw_sets = np.array(vec_sets, dtype=np.float32)
        raw_out = np.array(vec_out, dtype=np.float32)
        raw_workers = np.array(workers, dtype=np.float32)
        raw_machines = np.array(machines, dtype=np.float32)

        misc = [
            len(s.customers),
            s.demand,
            s.total_throughput,
            self.env.now / self.plan.duration,
            1.0 if s.approve_job else 0.0,
            float(s.batches["number"]),
            float(s.workers.total_workers)
        ]
        raw_misc = np.array(misc, dtype=np.float32)

        # ---------------------------------------------------------
        # --- NEU: Micro-Trends berechnen ---
        # ---------------------------------------------------------
        current_backlog = len(s.customers)
        current_wip = np.sum(raw_dis) + np.sum(raw_cln) + np.sum(raw_ins) + np.sum(raw_rep) + np.sum(raw_fin_comp)

        # Differenz zum letzten Schritt
        delta_backlog = current_backlog - self._prev_backlog_micro
        delta_wip = current_wip - self._prev_wip_micro
        delta_raw = total_supplies - self._prev_raw_micro

        # Historie überschreiben
        self._prev_backlog_micro = current_backlog
        self._prev_wip_micro = current_wip
        self._prev_raw_micro = total_supplies

        micro_trends = np.array([
            delta_backlog,
            delta_wip,
            delta_raw
        ], dtype=np.float32)
        # ---------------------------------------------------------

        raw_state = np.concatenate([
            raw_queues,
            raw_dis, raw_cln, raw_ins, raw_rep, raw_fin_comp,
            raw_sets,
            raw_out,
            raw_workers, raw_machines,
            raw_misc
        ], dtype=np.float32)

        context = self.context_extractor.extract_context(self.env.now / (24 * 60))

        # Finale Konkatenation: Base State + Macro Context + Micro Trends
        full_state = np.concatenate([raw_state, context, micro_trends])

        # State Dimension aktualisieren, damit PPOConfig sich automatisch anpasst
        self.state_dim = len(full_state)

        return full_state