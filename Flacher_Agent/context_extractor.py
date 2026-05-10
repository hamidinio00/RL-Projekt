import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Dict


@dataclass
class HistoricalRecord:
    """Single historical record for context extraction"""
    timestamp: float
    cores_arrived: Dict[str, int]  # {'high': x, 'low': y}
    orders_completed: int
    quality_yield: float  # Actual yield rate
    utilization: float
    backlog: int


class ContextExtractor:
    """
    Extracts context vector z from historical data.
    Gibt nun direkt die 24 berechneten Features zurück, ohne sie zu verrauschen.
    """

    def __init__(self,
                 window_days: int = 30,
                 aggregation_hours: int = 1):

        self.window_days = window_days
        self.aggregation_hours = aggregation_hours

        # Historical data storage
        self.history = deque(maxlen=window_days * 24 // aggregation_hours)

        # Die Output-Dimension ist jetzt fest 24 (die Anzahl der handgemachten Features)
        self.output_dim = 24

    def update_history(self, record: HistoricalRecord):
        self.history.append(record)

    def extract_context(self, current_time: float) -> np.ndarray:
        if len(self.history) < 2:
            return np.zeros(self.output_dim, dtype=np.float32)

        # GIB DIREKT DIE FEATURES ZURÜCK!
        # Kein torch.no_grad() und kein untrained Network mehr.
        features = self._compute_features()
        return features

    def _compute_features(self) -> np.ndarray:
        # Convert history to arrays
        timestamps = np.array([r.timestamp for r in self.history])
        cores_high = np.array([r.cores_arrived['high'] for r in self.history])
        cores_low = np.array([r.cores_arrived['low'] for r in self.history])
        quality_yields = np.array([r.quality_yield for r in self.history])
        utilizations = np.array([r.utilization for r in self.history])
        backlogs = np.array([r.backlog for r in self.history])

        features = []

        # 1. Core arrival features (6)
        total_cores = cores_high + cores_low
        features.extend([
            np.mean(total_cores),
            np.std(total_cores),
            np.mean(cores_high / (total_cores + 1e-6)),
            self._compute_trend(total_cores),
            self._compute_autocorr(total_cores, lag=7),
            self._compute_autocorr(total_cores, lag=1),
        ])

        # 2. Quality features (4)
        features.extend([
            np.mean(quality_yields),
            np.std(quality_yields),
            self._compute_trend(quality_yields),
            self._compute_regime_change_score(quality_yields),
        ])

        # 3. Demand/backlog features (4)
        features.extend([
            np.mean(backlogs),
            np.max(backlogs),
            self._compute_trend(backlogs),
            np.mean(np.diff(backlogs) > 0),
        ])

        # 4. Utilization features (3)
        features.extend([
            np.mean(utilizations),
            np.std(utilizations),
            np.mean(utilizations > 0.8),
        ])

        # 5. Time-based features (3)
        current_hour = timestamps[-1] % 24
        current_day = (timestamps[-1] // 24) % 7
        features.extend([
            np.sin(2 * np.pi * current_hour / 24),
            np.cos(2 * np.pi * current_hour / 24),
            current_day / 6.0,
        ])

        # 6. Cross-features (4)
        features.extend([
            np.corrcoef(total_cores, backlogs)[0, 1] if len(total_cores) > 1 and np.std(total_cores) > 0 and np.std(
                backlogs) > 0 else 0,
            np.mean(total_cores) / (np.mean(utilizations) + 1e-6),
            np.std(cores_high) / (np.std(cores_low) + 1e-6),
            self._compute_volatility_ratio(total_cores, backlogs),
        ])

        return np.array(features, dtype=np.float32)

    def _compute_trend(self, series: np.ndarray) -> float:
        if len(series) < 2: return 0.0
        x = np.arange(len(series))
        try:
            slope = np.polyfit(x, series, 1)[0]
            return np.clip(slope, -1, 1)
        except:
            return 0.0

    def _compute_autocorr(self, series: np.ndarray, lag: int) -> float:
        if len(series) <= lag: return 0.0
        s_mean = np.mean(series)
        s_var = np.var(series)
        if s_var < 1e-6: return 0.0
        series = series - s_mean
        try:
            return np.correlate(series[:-lag], series[lag:])[0] / (s_var * len(series))
        except:
            return 0.0

    def _compute_regime_change_score(self, series: np.ndarray, window: int = 7) -> float:
        if len(series) < 2 * window: return 0.0
        recent = series[-window:]
        previous = series[-2 * window:-window]
        mean_change = abs(recent.mean() - previous.mean()) / (previous.std() + 1e-6)
        var_change = abs(recent.var() - previous.var()) / (previous.var() + 1e-6)
        score = 1 - np.exp(-0.5 * (mean_change + var_change))
        return np.clip(score, 0, 1)

    def _compute_volatility_ratio(self, series1: np.ndarray, series2: np.ndarray) -> float:
        vol1 = series1.std()
        vol2 = series2.std()
        if vol2 < 1e-6: return 1.0
        return np.clip(vol1 / vol2, 0.1, 10.0)