# ------------------------------
# Block 7: Repair Processes
# ------------------------------
import simpy
import numpy as np


class Repair(object):
    def __init__(self, env, shop):
        self.env = env
        self.shop = shop
        self.workers = self.shop.workers.worker_repairing
        self.machines = self.shop.res_mgr.get("repair_machines")

        # Prozesszeiten (gleichverteilt) je Qualitätslevel
        self.repair_time_params = {0: (20, 25), 1: (15, 20), 2: (10, 15)}

        # Reparaturwahrscheinlichkeiten je Qualitätslevel
        self.repair_probabilities = {0: 0.9, 1: 0.5, 2: 0.2}

        # Bruchwahrscheinlichkeiten NUR in der Reparatur
        self.repair_break_prob = {0: 0.05, 1: 0.03, 2: 0.01}

        for _ in range(self.shop.plan.machines_rep):
            self.env.process(self.repair())

    # ------------------------------
    # Helper
    # ------------------------------
    def _rng(self):
        return self.shop.rand_mgr.rng_repair if getattr(self.shop, "rand_mgr", None) else np.random.default_rng()

    def _sample_repair_time(self, quality_level: int) -> float:
        rng = self._rng()
        low, high = self.repair_time_params.get(quality_level, self.repair_time_params[1])
        return float(rng.uniform(low, high))

    def _should_repair(self, quality_level: int) -> bool:
        rng = self._rng()
        p = self.repair_probabilities.get(quality_level, 0.5)
        return rng.random() < p

    def _break_in_repair(self, part) -> bool:
        rng = self._rng()
        q_level = getattr(part, "quality", 1)
        p_break = self.repair_break_prob.get(q_level, 0.03)
        if rng.random() < p_break:
            self.shop.discard.put(part)
            self.shop.log_event(part, "repair_failed_breakcheck")
            return True
        return False

    # ------------------------------
    # Methods as Simpy Processes
    # ------------------------------
    def repair(self):
        while True:
            try:
                # 1. Teil holen (erstes Yield, verhindert Worker Blocking)
                part = yield self.shop.inspected.get()

                # 2. Worker anfordern
                yield self.workers.get(1)
                self.shop.workers.start_working("repairing")

                q_level = getattr(part, "quality", 1)

                # 3. Entscheidung: Reparieren oder Skip?
                if not self._should_repair(q_level):
                    # Routing ohne Reparatur
                    if part.type in ("S", "R", "P", "M"):
                        self.shop.repaired.put(part)
                        part.t_rep = self.env.now
                        self.shop.log_event(part, "repair_skipped_to_repaired")
                    elif part.type in ("E", "C"):
                        self.shop.finished.put(part)
                        part.t_fin = self.env.now
                        self.shop.log_event(part, "repair_skipped_to_finished")
                    else:
                        self.shop.repaired.put(part)  # Fallback

                    # Worker freigeben & Neustart
                    self.shop.workers.stop_working("repairing")
                    yield self.workers.put(1)
                    continue

                # 4. Reparatur durchführen
                with self.machines.request() as req_2:
                    yield req_2
                    duration = self._sample_repair_time(q_level)

                    if self.shop.logging_enabled:
                        self.shop.proc_times["repair"].append(duration)
                    yield self.env.timeout(duration)

                # Bruch während Reparatur?
                if self._break_in_repair(part):
                    self.shop.workers.stop_working("repairing")
                    yield self.workers.put(1)
                    continue

                # 5. Routing nach erfolgreicher Reparatur
                if part.type in ("S", "R", "P", "M"):
                    self.shop.repaired.put(part)
                    part.t_rep = self.env.now
                    self.shop.log_event(part, "repair_repaired")
                elif part.type in ("E", "C"):
                    self.shop.finished.put(part)
                    part.t_fin = self.env.now
                    self.shop.log_event(part, "repair_finished")
                else:
                    self.shop.repaired.put(part)
                    part.t_rep = self.env.now

                self.shop.revenue += 500

                # Qualitätsupdate (Reparatur verbessert Zustand)
                part.quality = 2
                if hasattr(part, "true_quality"):
                    part.true_quality = 2

                self.shop.workers.stop_working("repairing")
                yield self.workers.put(1)

            except GeneratorExit:
                break