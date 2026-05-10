# ------------------------------
# Block 2: Shop Floor & Main Control Logic
# ------------------------------
from inspect import cleandoc
import simpy
import statistics
import numpy as np
from matplotlib import pyplot as plt

from simulation.processes.disassemble import Disassemble
from simulation.processes.testing import TestFacility
from simulation.processes.cleaning import Cleaning
from simulation.processes.repair import Repair
from simulation.processes.assemble import Assemble
from simulation.external import CustomerService, Workers, Storage, Core

# ------------------------------
# Global Settings
# ------------------------------
SINGLE_MODE = True


# ------------------------------
# Planning Decisions (Configuration)
# ------------------------------
class Plan(object):
    def __init__(self):
        self.max_customers = 100
        self.duration = 43200  # Simulationsdauer in Minuten (z.B. 1 Monat)
        self.total_workers = 20

        # Maschinen-Kapazitäten
        self.workstation_dis = 5
        self.sand_blaster = 3
        self.washing_machine = 3
        self.drier = 3
        self.test_bench_inspec = 3
        self.metrology = 3
        self.inspection_bench = 3
        self.machines_rep = 3
        self.painting_booth = 3
        self.assembly_station = 3

        # Steuerungs-Modi
        self.mode_order = 1  # 1=Push, 0=Pull
        self.mode_production = 1  # 1=Alle, 0=Nur Gute
        self.mode_customer = 0  # 0=FIFO, 1=Priorität


# ------------------------------
# Main Class: ShopFloor
# ------------------------------
class ShopFloor(object):
    def __init__(self, env, shop_plan, logging_enabled: bool = False, rand_mgr=None):
        self.env = env
        self.plan = shop_plan
        self.logging_enabled = logging_enabled
        self.rand_mgr = rand_mgr

        # 1. Basic configuration
        self.duration = self.plan.duration
        self.max_customers = self.plan.max_customers
        self.mode_order = self.plan.mode_order
        self.mode_production = self.plan.mode_production
        self.mode_customer = self.plan.mode_customer
        self.total_workers = self.plan.total_workers

        # 2. Metrics (Reward Calculation Basis)
        self.cost = 0.0
        self.total_cost = 0.0
        self.revenue = 0.0
        self.total_revenue = 0.0
        self.co2 = 0
        self.energy = 0
        self.throughput = 0
        self.total_throughput = 10  # Startwert (Initialbestand)
        self.counter_good = 0
        self.counter_bad = 0
        self.num_discard = 0
        self.core_quality_average = 0
        self.elapsed_time = 0

        # 3. Stores and Queues
        self.supplies = simpy.Store(env)
        self.supplies_t = simpy.Store(env)
        self.buffer_q0 = simpy.Store(env)  # Low Quality
        self.buffer_q1 = simpy.Store(env)  # Medium Quality
        self.buffer_q2 = simpy.Store(env)  # High Quality
        self.core_queue = simpy.Store(env)

        # FilterStores erlauben gezielten Zugriff (z.B. nach ID oder Typ)
        self.disassembled = simpy.FilterStore(env)
        self.cleaned = simpy.FilterStore(env)
        self.inspected = simpy.FilterStore(env)
        self.repaired = simpy.FilterStore(env)
        self.finished = simpy.FilterStore(env)
        self.product_done = simpy.Store(env)
        self.discard = simpy.Store(env)

        # Initialbestand an fertigen Produkten (damit erste Kunden bedient werden können)
        for i in range(10):
            product = Core(10000 + i)
            self.product_done.put(product)

        # 4. Customers & Action Interface
        self.demand = 0
        self.customers = []  # Liste der Customer Objekte
        self.customer = simpy.Store(env)  # Store für Versand
        self.orders = []

        # RL Action Schnittstellen
        self.batches = {"time": 0, "number": 0}
        self.batch_size_from_action = 2  # Default safe value
        self.order_release_from_action = 0
        self.batch_strategy_from_action = 0

        # 5. Data Logging
        self.supply_times = []
        self.size_supply = []
        self.core_quality_list = []
        self.core_done_vec = []
        self.current_count = []

        # State Buffer für RL (39 Channels)
        self.vectors = {i: [] for i in range(1, 40)}
        self.event_log = []
        self.proc_times = {
            "inspect1": [], "inspect2": [], "dis_step1": [], "dis_step2": [],
            "repair": [], "paint": [], "galv": [], "assembly": [],
        }

        # 6. Events
        self.done = simpy.Event(env)
        self.new_order = simpy.Event(env)
        self.new_supplier = simpy.Event(env)
        self.approve_job = False
        self.new_batch = False

        # 7. Sub-Modules
        self.ctrl = Controller(env)
        self.res_mgr = ResourceManager(env, self.plan)

        # Hinweis: Diese Klassen müssen im Scope verfügbar sein!
        self.customer_service = CustomerService(env, self)
        self.storage = Storage(env, self)
        self.workers = Workers(env, self)

        self.testing = TestFacility(env, self)
        self.disassemble = Disassemble(env, self)
        self.cleaning = Cleaning(env, self)
        self.repair = Repair(env, self)
        self.assemble = Assemble(env, self)

        # 8. Start Processes
        self.env.process(self.update())
        self.env.process(self.terminate())
        self.env.process(self.prod_section())

        if self.logging_enabled:
            self.env.process(self.monitor_system())

    # ------------------------------
    # Methods
    # ------------------------------

    def get_snapshot(self):
        """Hilfsfunktion für Wrapper Reward Calculation"""
        return self.cost, self.revenue, self.total_throughput

    def prod_section(self):
        """
        Periodischer Trigger für RL-Agenten (alle 2000 Steps),
        falls keine Ereignisse eintreten.
        """
        while True:
            yield self.env.timeout(2000)
            # Zwingt den Agenten zur Entscheidung / zum 'Step'
            self.ctrl.pause_2()
            self.total_throughput += self.throughput
            self.throughput = 0

    def terminate(self):
        """
        Beendet die Simulation sauber nach Ablauf der Zeit.
        Optimiert: Kein Polling mehr jede Sekunde.
        """
        remaining = self.duration - self.env.now
        if remaining > 0:
            yield self.env.timeout(remaining)

        # Finalize Stats
        self.num_discard = len(self.discard.items)
        if self.core_quality_list:
            self.core_quality_average = statistics.mean(self.core_quality_list)

        if not self.done.triggered:
            self.done.succeed()

    def log_event(self, part, stage: str):
        if not self.logging_enabled:
            return

        try:
            core_id = getattr(part, "id", None)
            part_type = getattr(part, "type", "core")
            q = getattr(part, "quality", None)
            q_true = getattr(part, "true_quality", None)
        except Exception:
            core_id = None
            part_type = "unknown"
            q = None
            q_true = None

        self.event_log.append({
            "time": self.env.now,
            "core_id": core_id,
            "part_type": part_type,
            "stage": stage,
            "quality": q,
            "true_quality": q_true,
        })

    def monitor_system(self):
        """
        Logging für Graphen.
        ACHTUNG: Im RL Training deaktivieren (logging_enabled=False),
        sonst Memory Overflow!
        """
        while True:
            # 1. Pufferstände
            self.vectors[1].append(len(self.core_queue.items))
            self.vectors[2].append(len(self.disassembled.items))
            self.vectors[19].append(len(self.cleaned.items))
            self.vectors[7].append(len(self.repaired.items))
            self.vectors[18].append(len(self.inspected.items))
            self.vectors[23].append(len(self.customer.items))
            self.core_done_vec.append(len(self.product_done.items))

            # 2. Logistik
            self.vectors[8].append(self.demand)
            self.vectors[9].append(len(self.supplies.items))
            total_buffered = (len(self.buffer_q0.items) +
                              len(self.buffer_q1.items) +
                              len(self.buffer_q2.items))
            self.vectors[29].append(total_buffered)
            self.vectors[30].append(len(self.finished.items))
            self.vectors[31].append(self.total_throughput)

            # 3. Ressourcen
            if hasattr(self, "res_mgr"):
                self.vectors[20].append(self.res_mgr.get("washing_machine").count)
                self.vectors[21].append(self.res_mgr.get("sand_blaster").count)
                self.vectors[22].append(self.res_mgr.get("oven").count)
                self.vectors[34].append(self.res_mgr.get("workstation").count)
                self.vectors[36].append(self.res_mgr.get("repair_machines").count)
                self.vectors[37].append(self.res_mgr.get("assembly_station").count)
                self.vectors[38].append(self.res_mgr.get("test_bench").count)

            # 4. Worker Stats
            if hasattr(self, "workers"):
                self.vectors[32].append(self.workers.distribution[0])

            self.vectors[33].append(self.batches["number"])

            yield self.env.timeout(1)

    def update(self):
        """Regelmäßige Status-Checks und Kunden-Priorisierung"""
        while True:
            yield self.env.timeout(1)

            # Produktionsfreigabe
            if self.mode_production == 1:
                if (
                        self.demand > 0  # FIX: Explizite Prüfung
                        and len(self.customer.items) >= self.demand
                        and len(self.customers) > 0
                ):
                    self.approve_job = True

            # Kunden-Logik
            if len(self.customers) > 1:
                # Mode 1: Smart Order Sorting
                if self.mode_customer == 1:
                    ready = len(self.supplies.items)
                    done = len(self.product_done.items)
                    demands = [(c.demand, c) for c in self.customers]
                    sorted_demands = sorted(demands, key=lambda x: x[0], reverse=True)

                    # Strategie: Große Aufträge zuerst, wenn Lager voll
                    if done > sorted_demands[0][0]:
                        target = sorted_demands[0][1]
                    # Kleine zuerst, wenn Lager fast leer
                    elif (done + ready) < sorted_demands[-1][0]:
                        target = sorted_demands[-1][1]
                    else:
                        target = self.customers[0]  # Fallback

                    # Umstrukturierung der Warteschlange
                    if target != self.customers[0]:
                        self.customers.remove(target)
                        self.customers.insert(0, target)

                    self.demand = self.customers[0].demand

                # Mode 0: FIFO
                elif self.mode_customer == 0:
                    self.demand = self.customers[0].demand

            elif len(self.customers) == 1:
                self.demand = self.customers[0].demand


# ------------------------------
# Helpers
# ------------------------------

class ResourceManager(object):
    def __init__(self, env, plan):
        self.env = env
        self.names = [
            "workstation", "washing_machine", "sand_blaster", "oven",
            "test_bench", "inspection_bench", "metrology",
            "repair_machines", "painting_booth", "assembly_station",
        ]
        # Ressourcen initialisieren
        self.resources = {}
        # Mapping von Attribut-Namen im Plan zu Dictionary-Keys
        plan_map = {
            "workstation": plan.workstation_dis,
            "washing_machine": plan.washing_machine,
            "sand_blaster": plan.sand_blaster,
            "oven": plan.drier,
            "test_bench": plan.test_bench_inspec,
            "inspection_bench": plan.inspection_bench,
            "metrology": plan.metrology,
            "repair_machines": plan.machines_rep,
            "painting_booth": plan.painting_booth,
            "assembly_station": plan.assembly_station
        }

        for name, capacity in plan_map.items():
            self.resources[name] = simpy.Resource(self.env, capacity)

    def get_current_count(self):
        return [self.resources[n].count for n in self.names]

    def get(self, name):
        return self.resources[name]


class Controller(object):
    """
    SMDP Controller: Pausiert die Simulation, damit der RL-Agent
    eine Aktion wählen kann.
    """

    def __init__(self, env):
        self.env = env
        self.event = self.new_event()
        self.event_2 = self.new_event()

    def pause(self):
        # Trigger für Supplier/Events
        if not self.event.triggered:
            self.event.succeed()
            self.event = self.new_event()

    def pause_2(self):
        # Trigger für Zeitintervalle/Kunden
        if not self.event_2.triggered:
            self.event_2.succeed()
            self.event_2 = self.new_event()

    def new_event(self):
        return self.env.event()