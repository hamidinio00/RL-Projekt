# ------------------------------
# Block 3: Random Number Management
# ------------------------------
import numpy as np
import random

class RandomManager:
    """
    Verwaltet benannte Zufallsströme für Reproduzierbarkeit (RL).
    Jeder stochastische SimPy-Prozess bekommt seinen eigenen RNG-Stream.
    """

    def __init__(self, base_seed: int, config_id: int = 0, run_id: int = 0):
        # SeedSequence kombiniert Basis, Konfiguration und Run-Index
        # Garantiert, dass jede Episode im RL-Training deterministisch ist, wenn die run_id hochgezählt wird.
        ss = np.random.SeedSequence([base_seed, config_id, run_id])

        # 26 unabhängige Streams für die verschiedenen Prozesse
        child_seqs = ss.spawn(26)

        # --- Supply / Storage ---
        self.rng_supplier = np.random.default_rng(child_seqs[0])
        self.rng_storage_handle = np.random.default_rng(child_seqs[1])
        self.rng_storage_core_add = np.random.default_rng(child_seqs[2])

        # --- CustomerService ---
        self.rng_customer_new = np.random.default_rng(child_seqs[3])
        self.rng_customer_job_done = np.random.default_rng(child_seqs[4])
        self.rng_customer_transfer = np.random.default_rng(child_seqs[5])
        self.rng_customer_track = np.random.default_rng(child_seqs[6])

        # --- Testing / Inspection ---
        self.rng_testing_pre = np.random.default_rng(child_seqs[7])
        self.rng_testing_final = np.random.default_rng(child_seqs[8])
        self.rng_testing_components = np.random.default_rng(child_seqs[9])

        # --- Disassembly ---
        self.rng_dis_step1 = np.random.default_rng(child_seqs[10])
        self.rng_dis_step2 = np.random.default_rng(child_seqs[11])

        # --- Cleaning ---
        self.rng_clean_chem = np.random.default_rng(child_seqs[12])
        self.rng_clean_mech1 = np.random.default_rng(child_seqs[13])
        self.rng_clean_mech2 = np.random.default_rng(child_seqs[14])
        self.rng_clean_dry1 = np.random.default_rng(child_seqs[15])
        self.rng_clean_dry2 = np.random.default_rng(child_seqs[16])
        self.rng_clean_dry3 = np.random.default_rng(child_seqs[17])

        # --- Repair ---
        self.rng_repair = np.random.default_rng(child_seqs[18])

        # --- Assembly ---
        self.rng_assemble_paint_stator = np.random.default_rng(child_seqs[19])
        self.rng_assemble_paint_rotor = np.random.default_rng(child_seqs[20])
        self.rng_assemble_galv_material = np.random.default_rng(child_seqs[21])
        self.rng_assemble_galv_pulley = np.random.default_rng(child_seqs[22])
        self.rng_assemble_assemble = np.random.default_rng(child_seqs[23])

        # --- Globale Bruchlogik ---
        self.rng_break = np.random.default_rng(child_seqs[24])

        # --- Sonstiges / Reserve ---
        self.rng_misc = np.random.default_rng(child_seqs[25])

        # Python Random Fallback (falls Libraries os.random nutzen)
        self.py_random = random.Random(int(self.rng_misc.integers(0, 2**31 - 1)))