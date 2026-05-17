# hierarchical_learning/hierarchical_wrapper.py
import simpy
import numpy as np

from learning.shield import ProductionShield
from learning.simpy_shop_wrapper import SimpyShopWrapper
from learning.action import Action
from learning.context_extractor import HistoricalRecord


class HierarchicalWrapper(SimpyShopWrapper):
    """
    Erweitert den SimpyShopWrapper für Hierarchisches RL.
    Update: Keine künstlichen Masking- oder Hektik-Strafen mehr!
    Nutzt das intelligente Blame-Assignment für den Shield.
    """

    def __init__(self, plan, init_batches, use_implicit_batch=False):
        super().__init__(plan, init_batches, use_implicit_batch=use_implicit_batch)

        self.use_implicit_batch = use_implicit_batch
        self.current_manager_action = None
        self.worker_step_duration = 480

        self.shield = ProductionShield(config=self.plan)

    def reset(self):
        # Basis-Klasse ruft SimPy Reset auf (inklusive Mikro-Trends Reset)
        state = super().reset()
        self.current_manager_action = None
        return state

    def step_hierarchical(self, worker_action_dict, manager_action_dict=None):
        # --- 1. MANAGER UPDATE ---
        if manager_action_dict:
            self.current_manager_action = manager_action_dict

        if self.current_manager_action is None:
            # Fallback Action für den Start
            base_action = {'capacity': 0, 'release': 2, 'batch': 0}
            if self.use_implicit_batch:
                base_action.update({'prio_low': 0, 'prio_mid': 0, 'prio_high': 0})
            else:
                base_action.update({'strategy': 0})
            self.current_manager_action = base_action

        # --- 2. ACTION MERGING ---
        # Manager entscheidet Batch Größe
        batch_sizes = [5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30]
        bs_idx = int(self.current_manager_action['batch'])
        real_bs = batch_sizes[bs_idx] if bs_idx < len(batch_sizes) else 15

        # Manager entscheidet Batch Parameter
        p_low = int(self.current_manager_action.get('prio_low', 0))
        p_mid = int(self.current_manager_action.get('prio_mid', 0))
        p_hi = int(self.current_manager_action.get('prio_high', 0))
        strat = int(self.current_manager_action.get('strategy', 0))

        # Raw Action zusammenbauen
        raw_action = Action(
            prio_inspection=int(worker_action_dict['prio_insp']),
            prio_disassembly=int(worker_action_dict['prio_dis']),
            prio_cleaning=int(worker_action_dict['prio_clean']),
            prio_repair=int(worker_action_dict['prio_rep']),
            prio_assembly=int(worker_action_dict['prio_ass']),
            order_release=int(self.current_manager_action['release']),
            capacity_level=int(self.current_manager_action['capacity']),
            batch_size=real_bs,
            batch_strategy=strat,
            batch_prio_low=p_low,
            batch_prio_mid=p_mid,
            batch_prio_high=p_hi
        )

        # --- 3. SHIELDING & INTELLIGENT BLAME ---
        cap_add = max(0, min(int(raw_action.capacity_level), 5))
        new_total_workers = 20 + cap_add
        self.shop.workers.set_capacity(new_total_workers)

        prios = [raw_action.prio_inspection, raw_action.prio_disassembly,
                 raw_action.prio_cleaning, raw_action.prio_repair, raw_action.prio_assembly]
        total_prio = sum(prios) if sum(prios) > 0 else 1
        est_counts = [int((p / total_prio) * new_total_workers) for p in prios]

        correction = self.shield.protect(
            raw_action,
            current_labor_budget=new_total_workers,
            shop_state=self.shop,
            planned_allocation=est_counts
        )
        safe_action = correction.safe_action
        total_shield_penalty = correction.penalty

        manager_shield_penalty = 0.0
        worker_shield_penalty = 0.0

        if safe_action.order_release != raw_action.order_release:
            manager_shield_penalty += 0.5
            worker_shield_penalty = max(0.0, total_shield_penalty - 0.5)
        else:
            worker_shield_penalty = total_shield_penalty

        final_prios = [
            safe_action.prio_inspection, safe_action.prio_disassembly, safe_action.prio_cleaning,
            safe_action.prio_repair, safe_action.prio_assembly
        ]
        total_prio_final = sum(final_prios)
        if total_prio_final == 0: total_prio_final = 1

        counts = [int((p / total_prio_final) * new_total_workers) for p in final_prios]
        remainder = new_total_workers - sum(counts)
        prio_indices = np.argsort(final_prios)[::-1]
        for i in range(remainder):
            counts[prio_indices[i % 5]] += 1

        self.shop.workers.reallocate(counts)

        # Parameter an Simulation übergeben
        self.shop.order_release_from_action = int(safe_action.order_release)
        self.shop.batch_size_from_action = int(safe_action.batch_size)
        self.shop.batch_strategy_from_action = int(safe_action.batch_strategy)

        # SimPy Normalisierung Logik für Implizite Gewichte
        if self.use_implicit_batch:
            raw_w = np.array([float(safe_action.batch_prio_low),
                              float(safe_action.batch_prio_mid),
                              float(safe_action.batch_prio_high)])
            tot = np.sum(raw_w)
            probs = raw_w / tot if tot > 0 else np.array([0.33, 0.33, 0.33])
            self.shop.batch_weights_from_action = probs.tolist()

        # --- 4. SIMULATION RUN ---
        c_prev, r_prev, _ = self.shop.get_snapshot()
        prev_throughput = self.shop.total_throughput

        start_time = self.env.now
        timeout_evt = self.env.timeout(self.worker_step_duration)
        ext_evts = [self.shop.ctrl.event, self.shop.ctrl.event_2]

        try:
            finished_events = self.env.run(until=simpy.AnyOf(self.env, [timeout_evt] + ext_evts))
        except simpy.Interrupt:
            finished_events = []

        duration = self.env.now - start_time

        manager_needed = False
        for evt in ext_evts:
            if evt in finished_events:
                manager_needed = True
                break

        # --- 5. REWARD CALCULATION ---
        self._update_context_history()

        c_now, r_now, _ = self.shop.get_snapshot()
        variable_cost_delta = c_now - c_prev
        revenue_delta = r_now - r_prev

        fixed_labor_cost = self._calculate_labor_cost(new_total_workers, duration)
        fixed_machine_cost = self.get_machine_cost_per_minute() * duration

        true_profit = revenue_delta - variable_cost_delta - fixed_labor_cost - fixed_machine_cost
        controllable_profit = revenue_delta - variable_cost_delta - fixed_labor_cost

        p_raw, p_wip, p_done = self._calculate_inventory_penalties()

        # --- A. MANAGER REWARD ---
        scaled_profit = controllable_profit / 7500.0
        throughput_delta = self.shop.total_throughput - prev_throughput

        utilization_penalty = 0.0
        if len(self.shop.customers) > 5 and throughput_delta == 0:
            utilization_penalty = 2.0

        total_manager_penalty = p_raw + p_wip + p_done + (manager_shield_penalty * 2.0)

        manager_reward_step = scaled_profit - total_manager_penalty - utilization_penalty
        manager_reward_step = np.clip(manager_reward_step, -20.0, 20.0)

        # --- B. WORKER REWARD ---
        rew_throughput = throughput_delta * 1.0
        total_worker_penalty = p_wip + (worker_shield_penalty * 2.0)

        worker_reward_step = rew_throughput - total_worker_penalty
        worker_reward_step = np.clip(worker_reward_step, -5.0, 5.0)

        self._last_cost = c_now
        self._last_rev = r_now
        self.total_reward += manager_reward_step

        state = self.get_state_vector()
        done = (self.env.now >= self.plan.duration)

        info = {
            'manager_needed': manager_needed,
            'step_profit': true_profit,
            'duration': duration,
            'wip_count': (len(self.shop.disassembled.items) + len(self.shop.cleaned.items) +
                          len(self.shop.inspected.items) + len(self.shop.repaired.items)),
            'throughput': self.shop.total_throughput,
            'workers': new_total_workers,
            'worker_reward': worker_reward_step,
            'manager_reward': manager_reward_step,
            'shield_penalty': total_shield_penalty
        }

        return state, worker_reward_step, done, info

    def _update_context_history(self):
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

        total = orders_completed + discarded_new
        yield_rate = orders_completed / total if total > 0 else 0.8

        record = HistoricalRecord(
            timestamp=self.env.now / (24 * 60),
            cores_arrived={'high': arr_high, 'low': arr_low},
            orders_completed=orders_completed,
            quality_yield=yield_rate,
            utilization=1.0,
            backlog=len(self.shop.customers)
        )
        self.context_extractor.update_history(record)

    def _calculate_labor_cost(self, n_workers, duration):
        base_rate = 0.25
        overtime = 1.5
        if n_workers <= 20:
            rate = n_workers * base_rate
        else:
            rate = (20 * base_rate) + ((n_workers - 20) * base_rate * overtime)
        return rate * duration

    def _calculate_inventory_penalties(self):
        q_raw = len(self.shop.buffer_q0.items) + len(self.shop.buffer_q1.items) + len(self.shop.buffer_q2.items)
        p_raw = q_raw * 0.001

        q_done = len(self.shop.product_done.items)
        p_done = q_done * 0.01

        count_wip = (len(self.shop.disassembled.items) + len(self.shop.cleaned.items) +
                     len(self.shop.inspected.items) + len(self.shop.repaired.items))
        wip_cores = count_wip / 6.0

        limit = 120.0
        if wip_cores > limit:
            excess = wip_cores - limit
            p_wip_raw = (limit * 0.004) + (excess ** 2 * 0.005)
            p_wip = min(p_wip_raw, 10.0)
        else:
            p_wip = wip_cores * 0.004

        return p_raw, p_wip, p_done