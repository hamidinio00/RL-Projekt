# learning/utils.py
import numpy as np


class RunningMeanStd:
    """
    Normalisiert Inputs dynamisch (Online-Algorithmus nach Welford).
    Zwingt den State-Vektor in einen Bereich um 0.0 mit Std 1.0.
    """

    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        # Sicherheits-Check: Input muss numpy array sein
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count

        # M2 ist die Summe der quadratischen Abweichungen
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x):
        # Clipping erweitert auf -10 bis +10.
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10.0, 10.0)