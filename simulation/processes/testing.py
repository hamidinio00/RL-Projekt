# ------------------------------
# Block 8: Testing & Inspection
# ------------------------------
import numpy as np
import simpy


class TestFacility(object):
    def __init__(self, env, shop_):
        self.env = env
        self.shop = shop_
        self.workers = self.shop.workers.worker_inspection
        self.inspectors = self.workers

        self.discarded = simpy.Store(self.env)
        self.test_bench = self.shop.res_mgr.get("test_bench")
        self.inspection_bench = self.shop.res_mgr.get("inspection_bench")
        self.metrology = self.shop.res_mgr.get("metrology")

        self.final_input = simpy.Store(self.env)

        for _ in range(self.shop.plan.inspection_bench):
            self.env.process(self.pre_test_core())
            self.env.process(self.components())

    # ------------------------------
    # RNG-Helper
    # ------------------------------
    def _rng_pre(self):
        return self.shop.rand_mgr.rng_testing_pre if getattr(self.shop, "rand_mgr", None) else np.random.default_rng()

    def _rng_final(self):
        return self.shop.rand_mgr.rng_testing_final if getattr(self.shop, "rand_mgr", None) else np.random.default_rng()

    def _rng_components(self):
        return self.shop.rand_mgr.rng_testing_components if getattr(self.shop, "rand_mgr", None) else np.random.default_rng()

    # ------------------------------
    # Methods as Simpy Processes
    # ------------------------------
    def pre_test_core(self):
        while True:
            try:
                # 1. Core holen (FIX: Zuerst Resource, dann Worker)
                core = yield self.shop.supplies.get()

                # 2. Worker holen
                yield self.inspectors.get(1)
                self.shop.workers.start_working("inspection")

                # 3. Test durchführen
                with self.test_bench.request() as req_2:
                    yield req_2
                    rng = self._rng_pre()
                    dur = self.duration(rng)
                    if self.shop.logging_enabled:
                        self.shop.proc_times["inspect1"].append(dur)
                    yield self.env.timeout(dur)

                # 4. Ergebnis auswerten & weiterleiten
                self.result(core, rng)
                self.shop.core_quality_list.append(core.quality)
                self.shop.log_event(core, "pre_test_core_end")
                # --- Routing in Quality-Buffer ---
                if core.quality == 2:
                    self.shop.buffer_q2.put(core)
                elif core.quality == 1:
                    self.shop.buffer_q1.put(core)
                else:
                    self.shop.buffer_q0.put(core)

                # 5. Worker freigeben
                self.shop.workers.stop_working("inspection")
                yield self.inspectors.put(1)

            except GeneratorExit:
                break

    def components(self):
        while True:
            parts = [0, 0, 0, 0, 0]

            try:
                # 1. Teile holen (FIX: Alle Teile sammeln, bevor Worker blockiert wird)
                # Risiko: Wenn ein Teil fehlt (z.B. verschrottet), wartet der Prozess hier ewig.
                stator = yield self.shop.cleaned.get(lambda s: s.type == "S")
                parts[0] = stator
                rotor = yield self.shop.cleaned.get(lambda s: s.type == "R")
                parts[1] = rotor
                pulley = yield self.shop.cleaned.get(lambda s: s.type == "P")
                parts[2] = pulley
                casting = yield self.shop.cleaned.get(lambda s: s.type == "C")
                parts[3] = casting
                electronics = yield self.shop.cleaned.get(lambda s: s.type == "E")
                parts[4] = electronics

                # 2. Inspektor belegen
                yield self.inspectors.get(1)
                self.shop.workers.start_working("inspection")

                with self.metrology.request() as req_2:
                    yield req_2
                    rng = self._rng_components()
                    dur = self.duration(rng)
                    if self.shop.logging_enabled:
                        self.shop.proc_times["inspect2"].append(dur)
                    yield self.env.timeout(dur)

                # 3. Entscheidungslogik für jedes Teil
                for part in parts:
                    decision_q = getattr(part, "true_quality", getattr(part, "quality", 1))

                    if decision_q == 0:
                        acc_prob = 0.3
                    elif decision_q == 1:
                        acc_prob = 0.6
                    elif decision_q == 2:
                        acc_prob = 0.9
                    else:
                        acc_prob = 0.6

                    if rng.random() < acc_prob:
                        if hasattr(part, "true_quality"):
                            part.quality = part.true_quality
                        self.shop.log_event(part, "inspection2_accept")
                        self.shop.inspected.put(part)
                    else:
                        self.shop.log_event(part, "inspection2_reject")
                        self.discarded.put(part)

                # 4. Inspektor freigeben
                self.shop.workers.stop_working("inspection")
                yield self.inspectors.put(1)

            except GeneratorExit:
                break

    # ------------------------------
    # Helper
    # ------------------------------
    def result(self, core_, rng):
        # zugrundeliegende wahre Qualität simulieren
        q_val = float(rng.normal(loc=0, scale=1))

        if q_val <= -0.6:
            true_quality = 0
        elif q_val <= 1:
            true_quality = 1
        else:
            true_quality = 2

        core_.update_true_quality(true_quality)

        # Beobachtete Qualität mit Messfehler
        if rng.random() < 0.8:
            observed_quality = true_quality
        else:
            # Wähle falsche Qualität
            other = [0, 1, 2]
            if true_quality in other:
                other.remove(true_quality)

            if other:
                observed_quality = int(rng.choice(other))
            else:
                observed_quality = true_quality

        core_.update_quality(observed_quality)

    def duration(self, rng):
        return int(rng.integers(20, 41))