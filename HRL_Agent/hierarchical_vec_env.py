# hierarchical_learning/hierarchical_vec_env.py
import multiprocessing as mp
import numpy as np
from simulation.internal import Plan
from hierarchical_learning.hierarchical_wrapper import HierarchicalWrapper


def worker_process(remote, parent_remote, plan, init_batch, use_implicit):
    """
    Der Prozess, der EINE Simulation hält.
    Nimmt 'use_implicit' entgegen, um den Wrapper korrekt zu konfigurieren.
    """
    parent_remote.close()

    # Jede Env braucht ihren eigenen Seed!
    # Wir setzen den Seed basierend auf der Prozess-ID implizit durch numpy random state
    np.random.seed(None)

    # HIER: Flag an den Wrapper übergeben
    env = HierarchicalWrapper(plan, init_batch, use_implicit_batch=use_implicit)

    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                w_action, m_action = data
                state, reward, done, info = env.step_hierarchical(w_action, m_action)
                if done:
                    # Auto-Reset bei Done
                    state = env.reset()
                remote.send((state, reward, done, info))
            elif cmd == 'reset':
                state = env.reset()
                remote.send(state)
            elif cmd == 'close':
                remote.close()
                break
            else:
                raise NotImplementedError
        except EOFError:
            break


class SubprocVecEnv:
    """
    Klasse zur Verwaltung mehrerer paralleler Simulationen.
    """

    def __init__(self, n_envs, plan, init_batch, use_implicit=False):
        """
        :param use_implicit: Steuert, ob die Envs im impliziten Modus (Prioritäten) starten.
        """
        self.n_envs = n_envs
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(n_envs)])
        self.ps = []

        for work_remote, remote in zip(self.work_remotes, self.remotes):
            # Arguments für den Worker Prozess packen
            # WICHTIG: use_implicit mitgeben
            p = mp.Process(target=worker_process, args=(work_remote, remote, plan, init_batch, use_implicit))
            p.daemon = True  # Prozesse sterben, wenn Main stirbt
            p.start()
            self.ps.append(p)
            work_remote.close()

    def step(self, w_actions, m_actions):
        """
        Führt Steps in allen Envs aus.
        w_actions: Liste von Dicts (Länge n_envs)
        m_actions: Liste von Dicts (Länge n_envs)
        """
        for remote, w_act, m_act in zip(self.remotes, w_actions, m_actions):
            remote.send(('step', (w_act, m_act)))

        results = [remote.recv() for remote in self.remotes]

        # Entpacken (Zip -> separate Listen)
        states, rewards, dones, infos = zip(*results)

        # Konvertierung zu Arrays für einfache Verarbeitung
        return np.stack(states), np.array(rewards), np.array(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()