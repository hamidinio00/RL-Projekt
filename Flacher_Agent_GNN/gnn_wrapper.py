import gymnasium as gym
import numpy as np
import torch
from torch_geometric.data import Data
import sys
import os

# Pfad-Setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class GraphObservationWrapper(gym.Env):
    """
    Dient als Adapter (Interface) zwischen der SimPy-Simulation und dem Gymnasium-Standard.
    Erbt direkt von gym.Env (statt Wrapper), da die darunterliegende SimPy-Klasse kein Gym-Env ist.
    """
    metadata = {"render_modes": []}

    def __init__(self, env, max_time_norm=20000.0):
        self.env = env
        self.max_time_norm = max_time_norm

        self.proc_ids = {
            "entry": 0, "disassemble": 1, "disassemble_step2": 1.5,
            "cleaning_chem": 2.1, "cleaning_mech": 2.2, "cleaning_dry": 2.3,
            "inspection": 3, "repair": 4, "painting": 5.1, "galvanizing": 5.2,
            "assembly_final": 5.9, "shipping": 6
        }

        # --- NEU: Mapping für One-Hot-Encoding ---
        self.num_classes = len(self.proc_ids)  # 12
        self.pid_to_idx = {pid: i for i, pid in enumerate(self.proc_ids.values())}

        # 4 Basis-Features + 12 One-Hot-Klassen = 16
        self.node_feat_dim = 4 + self.num_classes

        self.action_space = None
        self.observation_space = None

    def reset(self, seed=None, options=None):
        """
        Gymnasium Reset: Muss (obs, info) zurückgeben.
        """
        # SimPy Reset aufrufen
        # Da SimpyShopWrapper kein seed/options kennt, rufen wir es ohne auf
        obs = self.env.reset()

        # Adapter Logik: Falls das Env schon (obs, info) zurückgibt
        info = {}
        if isinstance(obs, tuple) and len(obs) == 2:
            obs, info = obs

        return self._get_graph_obs(), info

    def step(self, action):
        """
        Gymnasium Step: Muss (obs, reward, terminated, truncated, info) zurückgeben.
        """
        # 1. Simulation weiterschalten
        step_result = self.env.step(action)

        # 2. Adapter Logik: 4 vs 5 Werte behandeln
        if len(step_result) == 4:
            obs, reward, done, info = step_result
            terminated = done
            truncated = False  # SimPy läuft meist bis zum Ende
        else:
            obs, reward, terminated, truncated, info = step_result

        # 3. Graphen bauen
        graph_obs = self._get_graph_obs()

        # 4. Werte zurückgeben (Return war in deinem Snippet weg!)
        return graph_obs, reward, terminated, truncated, info

    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()

    def _get_graph_obs(self):
        shop = self.env.shop
        current_time = shop.env.now

        node_features = []
        edge_sources = []
        edge_targets = []

        curr_node_idx = 0
        machine_indices = {}

        # Hilfsfunktion für One-Hot
        def get_one_hot(pid_val):
            vec = [0.0] * self.num_classes
            if pid_val in self.pid_to_idx:
                vec[self.pid_to_idx[pid_val]] = 1.0
            return vec

        # =========================================================
        # A. MASCHINEN-KNOTEN
        # =========================================================
        res_config = {
            "workstation": ("disassemble", self.proc_ids["disassemble"]),
            "washing_machine": ("washing", self.proc_ids["cleaning_chem"]),
            "sand_blaster": ("blasting", self.proc_ids["cleaning_mech"]),
            "oven": ("drying", self.proc_ids["cleaning_dry"]),
            "test_bench": ("test_pre", self.proc_ids["inspection"]),
            "metrology": ("test_final", self.proc_ids["inspection"]),
            "inspection_bench": ("inspection_bench", self.proc_ids["inspection"]),
            "repair_machines": ("repair", self.proc_ids["repair"]),
            "painting_booth": ("painting", self.proc_ids["painting"]),
            "assembly_station": ("assembly", self.proc_ids["assembly_final"])
        }

        for res_name, (logic_name, pid) in res_config.items():
            if hasattr(shop, 'res_mgr'):
                resource = shop.res_mgr.get(res_name)
                capacity = max(1, resource.capacity)
                busy_ratio = resource.count / capacity
                queue_len_norm = len(resource.queue) / 10.0

                # NEU: [Type=1, Busy, QLen, 0.0] + [One-Hot-Vector (12 Dim)]
                base_feat = [1.0, busy_ratio, queue_len_norm, 0.0]
                one_hot_feat = get_one_hot(pid)

                node_features.append(base_feat + one_hot_feat)
                machine_indices[logic_name] = curr_node_idx
                curr_node_idx += 1

        # =========================================================
        # B. JOB-KNOTEN
        # =========================================================
        stores_to_check = []
        stores_to_check.append((shop.buffer_q0, "test_pre"))  # Korrektur: Geht erst in den Pre-Test
        stores_to_check.append((shop.buffer_q1, "test_pre"))
        stores_to_check.append((shop.buffer_q2, "test_pre"))
        stores_to_check.append((shop.core_queue, "disassemble"))

        if hasattr(shop, 'disassemble'):
            stores_to_check.append((shop.disassemble.further, "disassemble"))

        stores_to_check.append((shop.disassembled, "washing"))

        if hasattr(shop, 'cleaning'):
            stores_to_check.append((shop.cleaning.chem_cleaned, "blasting"))
            stores_to_check.append((shop.cleaning.mech_cleaned, "drying"))
            stores_to_check.append((shop.cleaning.dried_cleaned, "blasting"))

        stores_to_check.append((shop.cleaned, "test_final"))
        stores_to_check.append((shop.inspected, "repair"))
        stores_to_check.append((shop.repaired, "painting"))

        if hasattr(shop, 'assemble'):
            stores_to_check.append((shop.assemble.finished, "assembly"))

        stores_to_check.append((shop.finished, "assembly"))

        for store, target_key in stores_to_check:
            if target_key not in machine_indices:
                continue

            target_idx = machine_indices[target_key]
            items = []
            if hasattr(store, 'items'):
                items = store.items

            for job in items:
                quality = getattr(job, "quality", 0)
                start_time = 0
                if hasattr(job, 'casting'):
                    start_time = getattr(job.casting, "time", 0)
                elif hasattr(job, 'time'):
                    start_time = job.time

                if start_time == 0: start_time = current_time
                age_norm = np.clip((current_time - start_time) / self.max_time_norm, 0, 1)

                pid_feat = 0.0
                for r_name, (l_name, l_pid) in res_config.items():
                    if l_name == target_key:
                        pid_feat = float(l_pid)
                        break

                # HIER KORRIGIERT: [Type=0, Quality, Age, 1.0] + [One-Hot-Vector (12 Dim)]
                base_feat = [0.0, float(quality), age_norm, 1.0]
                one_hot_feat = get_one_hot(pid_feat)

                node_features.append(base_feat + one_hot_feat)

                # Edges (Job <-> Maschine)
                edge_sources.append(curr_node_idx)
                edge_targets.append(target_idx)
                edge_sources.append(target_idx)
                edge_targets.append(curr_node_idx)

                curr_node_idx += 1

        # =========================================================
        # B.2. MASCHINEN VERNETZEN (Materialfluss-Topologie)
        # =========================================================
        machine_flow = [
            # 1. Eingang -> Test -> Demontage (Nutzt dieselbe Workstation für Step 1 & 2)
            ("test_pre", "disassemble"),

            # 2. Demontage -> Chemische Reinigung (Der Hauptstrom)
            ("disassemble", "washing"),

            # 3. Flüsse aus der Chemischen Reinigung:
            # Material geht direkt in Reparatur (wenn nicht Ausschuss)
            ("washing", "repair"),

            # Stator/Casting gehen zur Mech1 (Sandstrahlen)
            ("washing", "blasting"),
            # Rotor/Pulley gehen zur Dry1 (Trocknen vor Mech2)
            ("washing", "drying"),

            # 4. Flüsse aus dem Trocknen/Strahlen (Kreuzungen)
            # Mech1 -> Dry2
            ("blasting", "drying"),
            # Dry1 -> Mech2
            ("drying", "blasting"),

            # 5. Alles Gereinigte geht zur Komponenten-Inspektion (Metrology)
            # (Die Elektronik kommt direkt aus Dry3)
            ("drying", "test_final"),
            ("blasting", "test_final"),
            ("washing", "test_final"),  # Fallback für den Fall, dass etwas direkt weitergeht

            # 6. Inspektion -> Reparatur
            ("test_final", "repair"),

            # 7. Reparatur -> Montagevorbereitung (Lackieren/Galvanisieren)
            # M, P, S, R gehen in die Painting Booth
            ("repair", "painting"),

            # 8. Endmontage (Führt alles zusammen)
            # S, R, P, M kommen aus der Painting Booth
            ("painting", "assembly"),
            # C, E kommen direkt von Reparatur (bzw. finished Store)
            ("repair", "assembly")
        ]

        # Kanten (Edges) in den Graphen einfügen
        for src_name, tgt_name in machine_flow:
            if src_name in machine_indices and tgt_name in machine_indices:
                s_idx = machine_indices[src_name]
                t_idx = machine_indices[tgt_name]

                # Bidirektionale Kanten
                edge_sources.append(s_idx)
                edge_targets.append(t_idx)
                edge_sources.append(t_idx)
                edge_targets.append(s_idx)

        # =========================================================
        # C. TENSOR BAUEN
        # =========================================================
        x = torch.tensor(node_features, dtype=torch.float)

        if len(edge_sources) > 0:
            edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index)
        data.num_nodes = curr_node_idx

        return data