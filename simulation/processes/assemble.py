# ------------------------------
# Block 4: Assembly Processes
# ------------------------------
import simpy
import numpy as np
from simulation.external import Core


class Assemble(object):
    def __init__(self, env, shop):
        self.env = env
        self.shop = shop

        # Resources
        self.workers = self.shop.workers.worker_assembling
        self.painting_booth = self.shop.res_mgr.get("painting_booth")
        self.assembly_station = self.shop.res_mgr.get("assembly_station")

        # Output Store
        self.finished = simpy.FilterStore(self.env)

        # Parameters
        self.paint_time_min = 25
        self.paint_time_max = 35
        self.galv_time_min = 25
        self.galv_time_max = 35

        # Start parallel processes based on capacity
        for _ in range(self.shop.plan.painting_booth):
            self.env.process(self.paint_stator())
            self.env.process(self.paint_rotor())
            self.env.process(self.galv_material())
            self.env.process(self.galv_pulley())

        # Assembly process (Main line)
        self.env.process(self.assemble())

    # ------------------------------
    # Sub-Processes (Painting / Galvanizing)
    # ------------------------------

    def paint_stator(self):
        while True:
            try:
                # 1. Teil holen
                stator = yield self.shop.repaired.get(lambda s: s.type == "S")

                # 2. Arbeiter holen
                yield self.workers.get(1)
                self.shop.workers.start_working("assembling")

                # 3. Maschine & Prozess
                with self.painting_booth.request() as req:
                    yield req

                    rng = self.shop.rand_mgr.rng_assemble_paint_stator if getattr(self.shop, "rand_mgr",
                                                                                  None) else np.random.default_rng()
                    duration = rng.uniform(self.paint_time_min, self.paint_time_max)

                    if self.shop.logging_enabled:
                        self.shop.proc_times["paint"].append(duration)

                    yield self.env.timeout(duration)

                # 4. Abschluss
                self.finished.put(stator)
                stator.t_fin = self.env.now
                self.shop.log_event(stator, "paint_stator")

                self.shop.workers.stop_working("assembling")
                yield self.workers.put(1)

            except GeneratorExit:
                break

    def paint_rotor(self):
        while True:
            try:
                rotor = yield self.shop.repaired.get(lambda s: s.type == "R")

                yield self.workers.get(1)
                self.shop.workers.start_working("assembling")

                with self.painting_booth.request() as req:
                    yield req
                    rng = self.shop.rand_mgr.rng_assemble_paint_rotor if getattr(self.shop, "rand_mgr",
                                                                                 None) else np.random.default_rng()
                    duration = rng.uniform(self.paint_time_min, self.paint_time_max)

                    if self.shop.logging_enabled:
                        self.shop.proc_times["paint"].append(duration)

                    yield self.env.timeout(duration)

                self.finished.put(rotor)
                rotor.t_fin = self.env.now
                self.shop.log_event(rotor, "paint_rotor")

                self.shop.workers.stop_working("assembling")
                yield self.workers.put(1)

            except GeneratorExit:
                break

    def galv_material(self):
        while True:
            try:
                material = yield self.shop.repaired.get(lambda s: s.type == "M")

                yield self.workers.get(1)
                self.shop.workers.start_working("assembling")

                rng = self.shop.rand_mgr.rng_assemble_galv_material if getattr(self.shop, "rand_mgr",
                                                                               None) else np.random.default_rng()
                duration = rng.uniform(self.galv_time_min, self.galv_time_max)

                if self.shop.logging_enabled:
                    self.shop.proc_times["galv"].append(duration)

                yield self.env.timeout(duration)

                self.finished.put(material)
                material.t_fin = self.env.now
                self.shop.log_event(material, "galv_material")

                self.shop.workers.stop_working("assembling")
                yield self.workers.put(1)

            except GeneratorExit:
                break

    def galv_pulley(self):
        while True:
            try:
                pulley = yield self.shop.repaired.get(lambda s: s.type == "P")

                yield self.workers.get(1)
                self.shop.workers.start_working("assembling")

                rng = self.shop.rand_mgr.rng_assemble_galv_pulley if getattr(self.shop, "rand_mgr",
                                                                             None) else np.random.default_rng()
                duration = rng.uniform(self.galv_time_min, self.galv_time_max)

                if self.shop.logging_enabled:
                    self.shop.proc_times["galv"].append(duration)

                yield self.env.timeout(duration)

                self.finished.put(pulley)
                pulley.t_fin = self.env.now
                self.shop.log_event(pulley, "galv_pulley")

                self.shop.workers.stop_working("assembling")
                yield self.workers.put(1)

            except GeneratorExit:
                break

    # ------------------------------
    # Final Assembly
    # ------------------------------
    def assemble(self):
        num = 0
        while True:
            # 1. Teile sammeln
            material = yield self.finished.get(lambda s: s.type == "M")
            pulley = yield self.finished.get(lambda s: s.type == "P")
            stator = yield self.finished.get(lambda s: s.type == "S")
            rotor = yield self.finished.get(lambda s: s.type == "R")

            electronics = yield self.shop.finished.get(lambda s: s.type == "E")
            casting = yield self.shop.finished.get(lambda s: s.type == "C")

            # 2. Worker & Ressource
            yield self.workers.get(1)
            self.shop.workers.start_working("assembling")

            try:
                with self.assembly_station.request() as req:
                    yield req

                    duration = 60.0  # Konstant

                    if self.shop.logging_enabled:
                        self.shop.proc_times["assembly"].append(duration)

                    yield self.env.timeout(duration)

                # 3. Produkt erstellen
                new_product = Core(num)
                num += 1

                new_product.stator = stator
                stator.t_ass = self.env.now
                new_product.rotor = rotor
                rotor.t_ass = self.env.now
                new_product.pulley = pulley
                pulley.t_ass = self.env.now
                new_product.electronics = electronics
                electronics.t_ass = self.env.now
                new_product.casting = casting
                casting.t_ass = self.env.now
                new_product.material = material
                material.t_ass = self.env.now

                self.shop.product_done.put(new_product)
                self.shop.revenue += 20000
                self.shop.throughput += 1

                self.shop.log_event(new_product, "assembly_finished")

                self.shop.workers.stop_working("assembling")
                yield self.workers.put(1)

            except GeneratorExit:
                break