from dataclasses import dataclass
@dataclass
class Action:
    """
    Definiert, wie eine Aktion des Agenten aussieht.
    Das ist reiner Datentransfer ohne Logik.
    """
    prio_disassembly: int
    prio_inspection: int
    prio_cleaning: int
    prio_repair: int
    prio_assembly: int

    order_release: int
    batch_size: int
    capacity_level: int
    batch_strategy: int
    batch_prio_low: float = 0.0
    batch_prio_mid: float = 0.0
    batch_prio_high: float = 0.0