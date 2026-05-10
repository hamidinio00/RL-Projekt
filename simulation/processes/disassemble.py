# ------------------------------
# Block 6: Disassembly Processes
# ------------------------------
import simpy
import numpy as np

# Sicherstellen, dass break_check importiert wird
from simulation.external import break_check


class Disassemble(object):
    def __init__(self, env, shop):
        self.env = env
        self.shop = shop
        self.workers = self.shop.workers.worker_disassemble
        self.workstation = self.shop.res_mgr.get("workstation")

        # ==========================================
        # PARAMETER DISASSEMBLY 1
        # ==========================================
        self.t_min_1 = 10.0
        self.t_max_1 = 90.0
        self.alpha_1 = 1.5
        modes_1 = {
            0: 37.0,  # Schlecht
            1: 25.0,  # Mittel
            2: 17.5,  # Gut
        }

        # Berechne Beta-Werte automatisch und speichere (alpha, beta, min, max)
        self.beta_params_step1 = {}
        for q, mode in modes_1.items():
            beta_val = self._calc_beta(self.t_min_1, self.t_max_1, mode, self.alpha_1)
            self.beta_params_step1[q] = (self.alpha_1, beta_val, self.t_min_1, self.t_max_1)

        # ==========================================
        # PARAMETER DISASSEMBLY 2
        # ==========================================
        self.t_min_2 = 10.0
        self.t_max_2 = 60.0
        self.alpha_2 = 1.5
        modes_2 = {
            0: 30.5,  # Schlecht
            1: 20.0,  # Mittel
            2: 14.5,  # Gut
        }

        # Berechne Beta-Werte automatisch und speichere (alpha, beta, min, max)
        self.beta_params_step2 = {}
        for q, mode in modes_2.items():
            beta_val = self._calc_beta(self.t_min_2, self.t_max_2, mode, self.alpha_2)
            self.beta_params_step2[q] = (self.alpha_2, beta_val, self.t_min_2, self.t_max_2)

        # Prozesse starten
        for _ in range(self.shop.plan.workstation_dis):
            self.env.process(self.disassemble_1())
            self.env.process(self.disassemble_2())

        self.further = simpy.Store(self.env)

    # ------------------------------
    # Helper: Beta-Berechnung nach Wurster et al. (2025)
    # ------------------------------
    def _calc_beta(self, t_min, t_max, t_mode, alpha):
        M = (t_mode - t_min) / (t_max - t_min)
        beta = ((alpha - 1) / M) - alpha + 2
        return beta

    # ------------------------------
    # Helper: Beta Sampling (Zieht die eigentliche Zeit)
    # ------------------------------
    def _sample_beta_step1(self, quality):
        rng = self.shop.rand_mgr.rng_dis_step1 if getattr(self.shop, "rand_mgr", None) else np.random.default_rng()
        # Fallback auf Qualität 1 (Mittel), falls ein unbekannter Wert kommt
        alpha, beta_param, min_t, max_t = self.beta_params_step1.get(quality, self.beta_params_step1[1])

        # Skaliere die 0-bis-1 Beta-Kurve auf deine echten Minuten (z.B. 10 bis 90)
        return min_t + rng.beta(alpha, beta_param) * (max_t - min_t)

    def _sample_beta_step2(self, quality):
        rng = self.shop.rand_mgr.rng_dis_step2 if getattr(self.shop, "rand_mgr", None) else np.random.default_rng()
        alpha, beta_param, min_t, max_t = self.beta_params_step2.get(quality, self.beta_params_step2[1])

        return min_t + rng.beta(alpha, beta_param) * (max_t - min_t)

    # ------------------------------
    # Step 1: Core -> C, M, P, E (+ Further)
    # ------------------------------
    def disassemble_1(self):
        while True:
            try:
                # FIX: Erst Core holen, dann Worker
                core = yield self.shop.core_queue.get()

                yield self.workers.get(1)
                self.shop.workers.start_working("disassemble")

                with self.workstation.request() as req_2:
                    yield req_2

                    real_quality = getattr(core, "true_quality", core.quality)
                    duration = self._sample_beta_step1(real_quality)

                    if self.shop.logging_enabled:
                        self.shop.proc_times["dis_step1"].append(duration)
                    yield self.env.timeout(duration)

                # Accounting
                self.shop.cost += 10
                self.shop.energy += 10
                self.shop.co2 += 5

                # Parts Distribution
                # Material
                if break_check(self.shop, core.material):
                    self.shop.disassembled.put(core.material)
                    core.material.t_dis = self.env.now

                # Casting
                if break_check(self.shop, core.casting):
                    self.shop.disassembled.put(core.casting)
                    core.casting.t_dis = self.env.now

                # Pulley
                if break_check(self.shop, core.pulley):
                    self.shop.disassembled.put(core.pulley)
                    core.pulley.t_dis = self.env.now

                # Electronics
                if break_check(self.shop, core.electronics):
                    self.shop.disassembled.put(core.electronics)
                    core.electronics.t_dis = self.env.now

                # Core für Schritt 2 (Rotor/Stator) vormerken
                self.further.put(core)

                self.shop.revenue += 500
                self.shop.log_event(core, "disassemble_step1")

                self.shop.workers.stop_working("disassemble")
                yield self.workers.put(1)

            except GeneratorExit:
                break

    # ------------------------------
    # Step 2: Further -> R, S
    # ------------------------------
    def disassemble_2(self):
        while True:
            try:
                # FIX: Erst Teil holen, dann Worker
                further = yield self.further.get()

                yield self.workers.get(1)
                self.shop.workers.start_working("disassemble")

                with self.workstation.request() as req_2:
                    yield req_2

                    real_quality = getattr(further, "true_quality", further.quality)
                    duration = self._sample_beta_step2(real_quality)

                    if self.shop.logging_enabled:
                        self.shop.proc_times["dis_step2"].append(duration)
                    yield self.env.timeout(duration)

                self.shop.cost += 10
                self.shop.energy += 10
                self.shop.co2 += 5

                # Rotor
                if break_check(self.shop, further.rotor):
                    self.shop.disassembled.put(further.rotor)
                    further.rotor.t_dis = self.env.now

                # Stator
                if break_check(self.shop, further.stator):
                    self.shop.disassembled.put(further.stator)
                    further.stator.t_dis = self.env.now

                self.shop.revenue += 500
                self.shop.log_event(further, "disassemble_step2")

                self.shop.workers.stop_working("disassemble")
                yield self.workers.put(1)

            except GeneratorExit:
                break