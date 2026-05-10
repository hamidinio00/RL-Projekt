# ------------------------------
# Block 1: External Logistics, People & Accounting
# ------------------------------
import simpy
import numpy as np
import statistics


# ------------------------------
# Supply Management
# ------------------------------

class Supplier(object):
    _id_counter = 1000

    def __init__(self, shop):
        self.shop = shop
        # RL-Seeding: Zugriff auf zentralen RNG Manager
        self.rng = shop.rand_mgr.rng_supplier if getattr(shop, "rand_mgr", None) else np.random.default_rng()

        # Liefermenge (Stochastik)
        self.number = max(1, int(self.rng.normal(loc=37, scale=15)))
        self.price = self.number * 3

        self.id = Supplier._id_counter
        Supplier._id_counter += 1000

    def delivery(self):
        return self.number


class Storage(object):
    def __init__(self, env, shop):
        self.env = env
        self.shop = shop
        self.mode_order = self.shop.plan.mode_order
        self.mode_production = self.shop.plan.mode_production

        self.env.process(self.handle_supplier())
        self.env.process(self.core_add())

    # ------------------------------
    # Deliveries including strategy
    # ------------------------------
    def handle_supplier(self):
        n = 0
        while True:
            # Basis-Taktung
            yield self.env.timeout(1)

            rng = self.shop.rand_mgr.rng_storage_handle if getattr(self.shop, "rand_mgr",
                                                                   None) else np.random.default_rng()

            # Mode 1: Periodische Bestellung (Push)
            if self.mode_order == 1:
                if self.env.now < 5:
                    # Initial supply
                    supplier = Supplier(self.shop)
                    self.shop.cost += supplier.price
                    id_ = supplier.id
                    for _ in range(supplier.number):
                        core = Core(id_)
                        id_ += 1
                        self.shop.supplies.put(core)

                # Normaler Zyklus
                raw_duration = int(rng.normal(loc=1500, scale=300))
                duration = max(1, raw_duration)
                self.shop.supply_times.append(duration)

                yield self.env.timeout(duration)

                new_supplier(self.shop, n)
                self.shop.new_batch = True
                n += 1

            # Mode 0: Pull-Strategie (Bestellung wenn leer)
            elif self.mode_order == 0:
                if (
                        len(self.shop.supplies_t.items) == 0
                        and len(self.shop.supplies.items) == 0
                        and len(self.shop.core_queue.items) == 0
                ):
                    new_supplier(self.shop, n)
                    n += 1

    # ------------------------------
    # Move from storage into the production (Interface to RL Action)
    # ------------------------------
    def core_add(self):
        while True:
            yield self.env.timeout(1)

            self.update_batches()
            batch = self.shop.batches
            batch_time = batch["time"]
            batch_size = batch["number"]
            strategy = getattr(self.shop, "batch_strategy_from_action", 0)
            weights = getattr(self.shop, "batch_weights_from_action", [0.33, 0.33, 0.33])

            if batch_size <= 0:
                continue

            # --- Warten auf Produktions-Start (Batch Time) ---
            if batch_time > 0:
                if self.mode_production == 1:
                    self.shop.approve_job = False
                elif self.mode_production == 0:
                    self.shop.approve_job = True
                yield self.env.timeout(batch_time)

            qualities_in_batch = set()

            # --- VORBEREITUNG IMPLICIT MODE (Strategy 99) ---
            targets = [0, 0, 0]
            if strategy == 99:
                w_sum = sum(weights)
                if w_sum > 0:
                    norm_weights = [w / w_sum for w in weights]
                    targets[0] = int(round(norm_weights[0] * batch_size))
                    targets[1] = int(round(norm_weights[1] * batch_size))
                    targets[2] = batch_size - targets[0] - targets[1]
                    if targets[2] < 0: targets[2] = 0
                else:
                    targets = [batch_size // 3, batch_size // 3, batch_size - 2 * (batch_size // 3)]

            taken_counts = [0, 0, 0]

            # --- ENTNAHME SCHLEIFE (Pro Teil im Batch) ---
            for _ in range(batch_size):
                core = None

                # -------------------------------------------------
                # A. IMPLIZITER MODUS (Strategy 99)
                # -------------------------------------------------
                if strategy == 99:
                    candidates = []

                    # Logik: Ziel noch nicht erreicht? Hoher Score. Sonst niedriger Score.

                    # Q0 (Low)
                    score_0 = -1
                    if len(self.shop.buffer_q0.items) > 0:
                        if taken_counts[0] < targets[0]:
                            score_0 = 100
                        else:
                            score_0 = 1
                    candidates.append((self.shop.buffer_q0, score_0, 0))

                    # Q1 (Mid)
                    score_1 = -1
                    if len(self.shop.buffer_q1.items) > 0:
                        if taken_counts[1] < targets[1]:
                            score_1 = 100
                        else:
                            score_1 = 1
                    candidates.append((self.shop.buffer_q1, score_1, 1))

                    # Q2 (High)
                    score_2 = -1
                    if len(self.shop.buffer_q2.items) > 0:
                        if taken_counts[2] < targets[2]:
                            score_2 = 100
                        else:
                            score_2 = 1
                    candidates.append((self.shop.buffer_q2, score_2, 2))

                    # Sortieren nach Score (Absteigend) -> Nimm Prio, sonst Lückenfüller
                    candidates.sort(key=lambda x: x[1], reverse=True)

                    for q, score, idx in candidates:
                        if score > -1:
                            core = yield q.get()
                            taken_counts[idx] += 1
                            break

                # -------------------------------------------------
                # EXPLIZITE MODI (0-3) - Wie bisher
                # -------------------------------------------------

                # Strategie 0: MIX (Nimm High -> Mid -> Low, oder warte auf egal was)
                if strategy == 0:
                    # Checke Verfügbarkeit (ohne yield, reiner Check)
                    if len(self.shop.buffer_q2.items) > 0:
                        core = yield self.shop.buffer_q2.get()
                    elif len(self.shop.buffer_q1.items) > 0:
                        core = yield self.shop.buffer_q1.get()
                    elif len(self.shop.buffer_q0.items) > 0:
                        core = yield self.shop.buffer_q0.get()
                    else:
                        # Wenn alle leer: Warte auf das erste Teil (Egal welche Q)
                        stores = [self.shop.buffer_q2, self.shop.buffer_q1, self.shop.buffer_q0]
                        events = [s.get() for s in stores]
                        winner = yield simpy.AnyOf(self.env, events)
                        for _, res in winner.items():
                            core = res
                            break

                # Strategie 1: NUR LOW (Quality 0)
                elif strategy == 1:
                    core = yield self.shop.buffer_q0.get()

                # Strategie 2: NUR MID (Quality 1)
                elif strategy == 2:
                    core = yield self.shop.buffer_q1.get()

                # Strategie 3: NUR HIGH (Quality 2)
                elif strategy == 3:
                    core = yield self.shop.buffer_q2.get()

                # Transfer zur Produktion
                if core:
                    qualities_in_batch.add(core.quality)
                    self.transfer(core)

            self.shop.new_batch = False

            # Statistik
            if len(qualities_in_batch) > 0 and hasattr(self.shop, 'batch_mix_stats'):
                is_mixed = 1 if len(qualities_in_batch) > 1 else 0
                self.shop.batch_mix_stats.append(is_mixed)

    def transfer(self, my_core):
        # Kerne in die Produktions-Queue schieben
        if my_core.quality in (0, 1, 2):
            my_core.put_init_time(self.env.now)
            self.shop.core_queue.put(my_core)
            self.shop.counter_good += 1
        else:
            # Nur ungültige/unbekannte Teile werden verworfen
            self.shop.discard.put(my_core)
            self.shop.counter_bad += 1

    def update_batches(self):
        # Verbindung zum RL-Agenten:
        self.shop.batches["time"] = self.shop.order_release_from_action * 60 * 24
        self.shop.batches["number"] = max(self.shop.batch_size_from_action, 2)
        # Fallback, falls Wrapper Variable noch nicht gesetzt hat
        if not hasattr(self.shop, 'batch_strategy_from_action'):
            self.shop.batch_strategy_from_action = 0


# ------------------------------
# Customers
# ------------------------------

class Customer(object):
    def __init__(self, name, demand, time, deadline):
        self.name = name
        self.demand = demand
        self.time_order = time
        self.deadline = deadline
        self.delivery = 0
        self.satisfied = False
        self.products = []

    def satisfactory(self):
        if self.delivery < self.deadline:
            self.satisfied = True

    def print(self):
        status = "satisfied" if self.satisfied else "not satisfied"
        print(f"{self.name} ordered {self.demand} products at {self.time_order}, "
              f"received at {self.delivery} -> {status}")


class CustomerService(object):
    def __init__(self, env, shop):
        self.env = env
        self.shop = shop
        self.env.process(self.new_customer())
        self.env.process(self.job_done())
        self.env.process(self.transfer_products())

    def new_customer(self):
        num = 1
        while True:
            rng = self.shop.rand_mgr.rng_customer_new if getattr(self.shop, "rand_mgr",
                                                                 None) else np.random.default_rng()

            # Zeit bis zum nächsten Kunden
            raw_time = int(rng.uniform(2880, 5040))
            time = max(1, raw_time)
            yield self.env.timeout(time)

            if len(self.shop.customers) < self.shop.max_customers:
                name = f"Customer_{num}"
                demand = abs(int(rng.normal(loc=75, scale=20))) + 1

                raw_deadline_offset = int(rng.normal(loc=43200, scale=2880))
                deadline_offset = max(1, raw_deadline_offset)
                deadline = self.env.now + deadline_offset

                new_customer = Customer(name, demand, self.env.now, deadline)
                self.shop.orders.append((demand, deadline - self.env.now))
                self.shop.customers.append(new_customer)
                num += 1

                # Update Stats & Pause for SMDP Decision (Neuer Auftrag = Entscheidungspunkt)
                self._update_and_pause_for_rl()

    def _update_and_pause_for_rl(self):
        """Helper to pause simulation and handover to RL Agent."""
        self.shop.elapsed_time = self.shop.env.now - self.shop.elapsed_time
        self.shop.total_cost += self.shop.cost
        self.shop.total_revenue += self.shop.revenue
        self.shop.total_throughput += self.shop.throughput
        # Hier triggert die Simulation einen Stop, damit der Agent entscheiden kann
        if hasattr(self.shop, 'ctrl'):
            self.shop.ctrl.pause_2()

    def job_done(self):
        while True:
            yield self.env.timeout(1)

            # shop.customer = Store für fertige Produkte
            # shop.customers = Liste der Kundenaufträge
            # shop.demand = Aktueller Bedarf des vordersten Kunden

            # Logik Korrektur: Reihenfolge und null-check
            if (self.shop.demand > 0
                    and len(self.shop.customers) > 0
                    and len(self.shop.customer.items) >= self.shop.demand
                    and self.shop.approve_job):

                current_customer = self.shop.customers[0]
                current_customer.delivery = self.env.now
                current_customer.satisfactory()

                #Tardiness History für den Benchmark tracken
                if not hasattr(self.shop, 'tardiness_history'):
                    self.shop.tardiness_history = []

                    # Tardiness = Lieferzeitpunkt - Deadline
                    # <= 0 bedeutet pünktlich/zu früh, > 0 bedeutet verspätet
                tardiness = self.env.now - current_customer.deadline
                self.shop.tardiness_history.append(tardiness)

                if current_customer.satisfied:
                    self.shop.total_revenue += 500

                # Produkte aus dem Store an den Kunden übergeben
                for _ in range(self.shop.demand):
                    prod = yield self.shop.customer.get()
                    prod.lead_time()
                    current_customer.products.append(prod)

                # remove the dispatched customer logic
                # Hinweis: vectors[10] ist hardcoded, sollte später verifiziert werden
                if len(self.shop.vectors) > 10:
                    self.shop.vectors[10].append(self.shop.customers.pop(0))
                else:
                    self.shop.customers.pop(0)

    def transfer_products(self):
        # Simuliert Transportlogistik / Versandvorbereitung
        while True:
            yield self.env.timeout(2500)
            # Lagerkosten für fertige Produkte
            self.shop.cost += 2500 * (len(self.shop.product_done.items)) * 0.5

            # Wenn Bedarf da ist, Produkte in den Versand-Store (shop.customer) schieben
            # Sonst bleiben sie in shop.product_done
            # HINWEIS: shop.customer sollte besser shop.shipping_dock heißen
            if len(self.shop.customer.items) < self.shop.demand:
                # Versuche alle fertigen Produkte zu verschieben
                # Achtung: Iteration über Items und get/put ist in SimPy tricky.
                # Besser: Einzeln prüfen.
                amount = len(self.shop.product_done.items)
                for _ in range(amount):
                    # Check erneut, da sich Bedarf geändert haben könnte
                    if len(self.shop.customer.items) < self.shop.demand:
                        prod = yield self.shop.product_done.get()
                        yield self.shop.customer.put(prod)
                    else:
                        break


# ------------------------------
# Core Components
# ------------------------------
# (Klassen unverändert übernommen, da reine Datencontainer)

class Core(object):
    def __init__(self, core_id):
        self.id = core_id
        self.quality = 0
        self.true_quality = 0
        self.total_lead_time = 0
        self.rotor = Rotor(core_id)
        self.stator = Stator(core_id)
        self.casting = Casting(core_id)
        self.pulley = Pulley(core_id)
        self.electronics = Electronics(core_id)
        self.material = Material(core_id)

    def update_quality(self, quality):
        self.quality = quality
        self.rotor.quality = quality
        self.stator.quality = quality
        self.casting.quality = quality
        self.pulley.quality = quality
        self.electronics.quality = quality
        self.material.quality = quality

    def update_true_quality(self, quality):
        self.true_quality = quality
        self.rotor.true_quality = quality
        self.stator.true_quality = quality
        self.casting.true_quality = quality
        self.pulley.true_quality = quality
        self.electronics.true_quality = quality
        self.material.true_quality = quality

    def put_init_time(self, time):
        self.casting.time = time
        self.stator.time = time
        self.rotor.time = time
        self.pulley.time = time
        self.electronics.time = time
        self.material.time = time

    def lead_time(self):
        # Berechnung der Durchlaufzeit beim Versand
        self.casting.time = self.casting.t_ass - self.casting.time
        self.stator.time = self.stator.t_ass - self.stator.time
        self.rotor.time = self.rotor.t_ass - self.rotor.time
        self.pulley.time = self.pulley.t_ass - self.pulley.time
        self.electronics.time = self.electronics.t_ass - self.electronics.time
        self.material.time = self.material.t_ass - self.material.time

        self.total_lead_time = (
                                       self.casting.time
                                       + self.stator.time
                                       + self.pulley.time
                                       + self.electronics.time
                                       + self.material.time
                                       + self.rotor.time
                               ) / 6


class Casting(object):
    def __init__(self, id_):
        self.id = id_
        self.quality = 0
        self.true_quality = 0
        self.time = 0
        self.type = "C"
        self.t_dis = 0
        self.t_clean = 0
        self.t_rep = 0
        self.t_ass = 0
        self.t_fin = 0


class Stator(object):
    def __init__(self, id_):
        self.id = id_
        self.quality = 0
        self.true_quality = 0
        self.time = 0
        self.type = "S"
        self.t_dis = 0
        self.t_clean = 0
        self.t_rep = 0
        self.t_ass = 0
        self.t_fin = 0


class Rotor(object):
    def __init__(self, id_):
        self.id = id_
        self.quality = 0
        self.true_quality = 0
        self.time = 0
        self.type = "R"
        self.t_dis = 0
        self.t_clean = 0
        self.t_rep = 0
        self.t_ass = 0
        self.t_fin = 0


class Pulley(object):
    def __init__(self, id_):
        self.id = id_
        self.quality = 0
        self.true_quality = 0
        self.time = 0
        self.type = "P"
        self.t_dis = 0
        self.t_clean = 0
        self.t_rep = 0
        self.t_ass = 0
        self.t_fin = 0


class Electronics(object):
    def __init__(self, id_):
        self.id = id_
        self.quality = 0
        self.true_quality = 0
        self.time = 0
        self.type = "E"
        self.t_dis = 0
        self.t_clean = 0
        self.t_rep = 0
        self.t_ass = 0
        self.t_fin = 0


class Material(object):
    def __init__(self, id_):
        self.id = id_
        self.quality = 0
        self.true_quality = 0
        self.time = 0
        self.type = "M"
        self.t_dis = 0
        self.t_clean = 0
        self.t_rep = 0
        self.t_ass = 0
        self.t_fin = 0


# ------------------------------
# Workers (Resource Management)
# ------------------------------

class Workers(object):
    def __init__(self, env, shop):
        self.env = env
        self.shop = shop

        self.MAX_POSSIBLE_WORKERS = 30

        # Sicherstellen, dass shop.total_workers gesetzt ist
        self.total_workers = getattr(self.shop.plan, 'total_workers', 20)  # Default Fallback
        self.shop.total_workers = self.total_workers

        # SimPy Container als Ressourcen-Pools mit MAX capacity
        self.pools = {
            "inspection": simpy.Container(env, capacity=self.MAX_POSSIBLE_WORKERS, init=0),
            "disassemble": simpy.Container(env, capacity=self.MAX_POSSIBLE_WORKERS, init=0),
            "cleaning": simpy.Container(env, capacity=self.MAX_POSSIBLE_WORKERS, init=0),
            "repairing": simpy.Container(env, capacity=self.MAX_POSSIBLE_WORKERS, init=0),
            "assembling": simpy.Container(env, capacity=self.MAX_POSSIBLE_WORKERS, init=0),
        }

        # Mapping für direkten Zugriff
        self.worker_inspection = self.pools["inspection"]
        self.worker_disassemble = self.pools["disassemble"]
        self.worker_cleaning = self.pools["cleaning"]
        self.worker_repairing = self.pools["repairing"]
        self.worker_assembling = self.pools["assembling"]

        # State Tracking
        self.working_count = {k: 0 for k in self.pools.keys()}
        self.utilization_log = {k: [] for k in self.pools.keys()}

        # Initial Allocation
        avg = self.total_workers // 5
        initial_counts = [avg] * 5
        # Rest verteilen
        for i in range(self.total_workers % 5):
            initial_counts[i] += 1

        self.current_allocation = {
            "inspection": initial_counts[0],
            "disassemble": initial_counts[1],
            "cleaning": initial_counts[2],
            "repairing": initial_counts[3],
            "assembling": initial_counts[4],
        }

        self.env.process(self._initial_fill(initial_counts))

        self.distribution = initial_counts
        # Diese Werte werden vom RL Agenten gesetzt (Action)
        self.from_action_disassembly = 4
        self.from_action_quality = 4

    def _initial_fill(self, counts):
        keys = ["inspection", "disassemble", "cleaning", "repairing", "assembling"]
        for i, key in enumerate(keys):
            yield self.pools[key].put(counts[i])

    def start_working(self, dept):
        self.working_count[dept] += 1
        # Für RL State: Loggen
        self.utilization_log[dept].append((self.env.now, self.working_count[dept]))

    def stop_working(self, dept):
        self.working_count[dept] -= 1
        self.utilization_log[dept].append((self.env.now, self.working_count[dept]))

    def set_capacity(self, new_total):
        """
        NEU: Wird vom Wrapper aufgerufen, wenn der Agent die Kapazität ändert.
        """
        target = max(20, min(new_total, 26))
        if target != self.total_workers:
            self.total_workers = target
            self.shop.total_workers = target

    def reallocate(self, target_counts: list):
        """
        Hauptfunktion für den RL-Agenten, um Arbeiter neu zu verteilen.
        """
        self.distribution = target_counts
        self.env.process(self._process_reallocation(target_counts))

    def _process_reallocation(self, target_counts):
        # Das aktuelle Limit kommt aus set_capacity
        current_limit = self.total_workers

        # 1. Normalisierung, falls Agent mehr verteilen will als erlaubt
        if sum(target_counts) > current_limit:
            # Proportionales Kürzen
            factor = current_limit / sum(target_counts)
            target_counts = [int(x * factor) for x in target_counts]

            # Auffüllen von Rundungsfehlern, bis Limit erreicht
            idx = 0
            while sum(target_counts) < current_limit:
                target_counts[idx % 5] += 1
                idx += 1

        # 2. Reallocation durchführen
        keys = ["inspection", "disassemble", "cleaning", "repairing", "assembling"]

        for i, key in enumerate(keys):
            target = target_counts[i]
            current = self.current_allocation[key]
            diff = target - current

            if diff > 0:
                # Mehr Arbeiter in den Pool geben
                yield self.pools[key].put(diff)
                self.current_allocation[key] += diff

            elif diff < 0:
                # Arbeiter abziehen
                yield self.pools[key].get(abs(diff))
                self.current_allocation[key] += diff


# ------------------------------
# Accounting & Helper
# ------------------------------

class Accounting(object):
    def __init__(self, env, shop):
        self.env = env
        self.shop = shop
        self.env.process(self.inventory_costs())

    def inventory_costs(self):
        while True:
            # Berechnung jede Minute
            yield self.env.timeout(1)

            # --- 1. Rohmaterial (Input Puffer) ---
            raw_count = (len(self.shop.core_queue.items) +
                         len(self.shop.buffer_q0.items) +
                         len(self.shop.buffer_q1.items) +
                         len(self.shop.buffer_q2.items))

            self.shop.cost += raw_count * 0.05

            # --- 2. Work in Progress (WIP) ---
            count_wip_parts = (
                    len(self.shop.disassembled.items) +
                    len(self.shop.cleaned.items) +
                    len(self.shop.inspected.items) +
                    len(self.shop.repaired.items) +
                    len(self.shop.finished.items)  # Komponenten die auf Montage warten (E/C)
            )

            # Umrechnung in Core-Äquivalente (6 Teile = 1 Core)
            wip_in_cores = count_wip_parts / 6.0
            self.shop.cost += wip_in_cores * 0.10

    def setup_cost(self):
        while True:
            yield self.env.timeout(1500)
            machine_cost(self.shop)

    def pay_workers(self):
        while True:
            yield self.env.timeout(1500)

            # Basis-Parameter
            base_rate = 15 * 25  # Kosten pro Worker pro Periode
            base_workers = 20  # Bis hier normaler Preis
            overtime_factor = 1.5  # 50% Aufschlag für Zusatzkräfte

            current_total = self.shop.total_workers

            if current_total <= base_workers:
                # Fall A: Normalkosten
                salary_cost = current_total * base_rate
            else:
                # Fall B: Basis + Overtime
                excess = current_total - base_workers
                salary_cost = (base_workers * base_rate) + (excess * base_rate * overtime_factor)

            self.shop.cost += salary_cost

def machine_cost(shop):
    # Kostenberechnung basierend auf Maschinenpark im Plan
    cost_map = {
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

    current_cost = 0
    for attr, cost in cost_map.items():
        count = getattr(shop.plan, attr, 0)
        current_cost += count * cost

    shop.cost += current_cost


def new_supplier(shop, n):
    supplier = Supplier(shop)
    shop.size_supply.append(supplier.number)
    shop.cost += supplier.price
    id_ = supplier.id
    for _ in range(supplier.number):
        core = Core(id_)
        id_ += 1
        shop.supplies.put(core)

    if n != 0:
        shop.num_discard = len(shop.discard.items)
        if shop.core_quality_list:
            shop.core_quality_average = statistics.mean(shop.core_quality_list)

        # RL State Info Update
        shop.current_count = shop.res_mgr.get_current_count() if hasattr(shop, 'res_mgr') else []

        shop.elapsed_time = shop.env.now - shop.elapsed_time
        shop.total_cost += shop.cost
        shop.total_throughput += shop.throughput

        # Trigger RL Pause (SMDP step)
        if hasattr(shop, 'ctrl'):
            shop.ctrl.pause()


def break_check(shop, part):
    """
    Entscheidet, ob ein Teil bei der Bearbeitung bricht.
    Nutzt 'true_quality' wenn vorhanden, sonst 'quality'.
    """
    rng = shop.rand_mgr.rng_break if getattr(shop, "rand_mgr", None) else np.random.default_rng()

    eff_q = getattr(part, "true_quality", getattr(part, "quality", 1))

    # Wahrscheinlichkeiten für Defekt/Bruch
    # Quality 1 (Mittel): 25% Bruchwahrscheinlichkeit (0.85 success)
    if eff_q == 1:
        if rng.random() > 0.75:
            shop.discard.put(part)
            return False
        return True

    # Quality 2 (Gut): 10% Bruchwahrscheinlichkeit (0.90 success)
    if eff_q == 2:
        if rng.random() > 0.90:
            shop.discard.put(part)
            return False
        return True

    # Quality 0 (Schlecht): 50% Bruch
    if eff_q == 0:
        if rng.random() > 0.5:
            shop.discard.put(part)
            return False
        return True

    return True