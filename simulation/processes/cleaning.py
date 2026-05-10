# ------------------------------
# Block 5: Cleaning Processes
# ------------------------------
import simpy
import numpy as np


class Cleaning(object):
    def __init__(self, env, shop):
        self.env = env
        self.shop = shop

        self.workers = self.shop.workers.worker_cleaning

        self.washing_machine = self.shop.res_mgr.get("washing_machine")
        self.sand_blaster = self.shop.res_mgr.get("sand_blaster")
        self.oven = self.shop.res_mgr.get("oven")

        # Output Stores
        self.chem_cleaned = simpy.FilterStore(self.env)
        self.mech_cleaned = simpy.FilterStore(self.env)
        self.dried_cleaned = simpy.FilterStore(self.env)

        # Prozesse starten
        for _ in range(self.shop.plan.washing_machine):
            self.env.process(self.chem())
            self.env.process(self.mech_1())
            self.env.process(self.mech_2())
            self.env.process(self.dry_1())
            self.env.process(self.dry_2())
            self.env.process(self.dry_3())

    # ------------------------------------------------------------------
    # Chemische Reinigung (Batch Process)
    # ------------------------------------------------------------------
    def chem(self):
        while True:
            # Warten auf komplettes Set (Risiko: Asymmetrie durch Ausschuss in Disassembly)
            try:
                stator = yield self.shop.disassembled.get(lambda s: s.type == "S")
                rotor = yield self.shop.disassembled.get(lambda s: s.type == "R")
                pulley = yield self.shop.disassembled.get(lambda s: s.type == "P")
                casting = yield self.shop.disassembled.get(lambda s: s.type == "C")
                electronics = yield self.shop.disassembled.get(lambda s: s.type == "E")
                material = yield self.shop.disassembled.get(lambda s: s.type == "M")

                # Hier kein Worker nötig
                with self.washing_machine.request() as req:
                    yield req
                    yield self.env.timeout(30)

                # --- Bruchlogik für Material M ---
                rng = self.shop.rand_mgr.rng_clean_chem if getattr(self.shop, "rand_mgr",
                                                                   None) else np.random.default_rng()

                # Wahre Qualität nutzen
                qual = getattr(material, "true_quality", getattr(material, "quality", 1))

                # Success Probabilities: Quality 2 (Good) -> 80%, 0 (Bad) -> 25%
                prob_map = {2: 0.80, 1: 0.60, 0: 0.25}
                prob_success = prob_map.get(qual, 0.40)

                if rng.random() < prob_success:
                    self.shop.repaired.put(material)
                else:
                    self.shop.discard.put(material)

                # --- Alle anderen Teile weiterleiten ---
                self.chem_cleaned.put(stator)
                self.chem_cleaned.put(rotor)
                self.chem_cleaned.put(pulley)
                self.chem_cleaned.put(casting)
                self.chem_cleaned.put(electronics)

                self.shop.revenue += 500

            except GeneratorExit:
                break

    # ------------------------------------------------------------------
    # Mechanische Reinigung 1 (Sandstrahlen Stator/Casting)
    # ------------------------------------------------------------------
    def mech_1(self):
        while True:
            try:
                # FIX: Erst Teile, dann Worker
                stator = yield self.chem_cleaned.get(lambda s: s.type == "S")
                casting = yield self.chem_cleaned.get(lambda s: s.type == "C")

                yield self.workers.get(1)
                self.shop.workers.start_working("cleaning")

                with self.sand_blaster.request() as req_2:
                    yield req_2
                    yield self.env.timeout(15)

                self.shop.cleaned.put(casting)
                casting.t_clean = self.env.now

                self.mech_cleaned.put(stator)
                self.shop.revenue += 500

                self.shop.workers.stop_working("cleaning")
                yield self.workers.put(1)

            except GeneratorExit:
                break

    # ------------------------------------------------------------------
    # Mechanische Reinigung 2 (Sandstrahlen Rotor/Pulley)
    # ------------------------------------------------------------------
    def mech_2(self):
        while True:
            try:

                rotor = yield self.dried_cleaned.get(lambda s: s.type == "R")
                pulley = yield self.dried_cleaned.get(lambda s: s.type == "P")

                yield self.workers.get(1)
                self.shop.workers.start_working("cleaning")

                with self.sand_blaster.request() as req_2:
                    yield req_2
                    yield self.env.timeout(15)

                self.shop.cleaned.put(rotor)
                rotor.t_clean = self.env.now

                self.shop.cleaned.put(pulley)
                pulley.t_clean = self.env.now

                self.shop.workers.stop_working("cleaning")
                yield self.workers.put(1)

            except GeneratorExit:
                break

    # ------------------------------------------------------------------
    # Trocknung 1 (Rotor/Pulley vor Mech2)
    # ------------------------------------------------------------------
    def dry_1(self):
        while True:
            try:
                # Hier kein Worker nötig laut Originalcode
                rotor = yield self.chem_cleaned.get(lambda s: s.type == "R")
                pulley = yield self.chem_cleaned.get(lambda s: s.type == "P")

                with self.oven.request() as req:
                    yield req
                    yield self.env.timeout(15)

                self.dried_cleaned.put(rotor)
                self.dried_cleaned.put(pulley)
                self.shop.revenue += 500
            except GeneratorExit:
                break

    # ------------------------------------------------------------------
    # Trocknung 2 (Stator nach Mech1)
    # ------------------------------------------------------------------
    def dry_2(self):
        while True:
            try:
                stator = yield self.mech_cleaned.get(lambda s: s.type == "S")
                with self.oven.request() as req:
                    yield req
                    yield self.env.timeout(15)

                self.shop.cleaned.put(stator)
                stator.t_clean = self.env.now
            except GeneratorExit:
                break

    # ------------------------------------------------------------------
    # Trocknung 3 (Electronics direkt nach Chem)
    # ------------------------------------------------------------------
    def dry_3(self):
        while True:
            try:
                electronics = yield self.chem_cleaned.get(lambda s: s.type == "E")
                with self.oven.request() as req:
                    yield req
                    yield self.env.timeout(15)

                self.shop.cleaned.put(electronics)
                electronics.t_clean = self.env.now
            except GeneratorExit:
                break