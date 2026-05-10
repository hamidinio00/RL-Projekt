# learning/vec_env.py
import multiprocessing as mp
import numpy as np
import traceback
from learning.action import Action
from learning.simpy_shop_wrapper import SimpyShopWrapper

# Befehls-Codes
CMD_RESET = 0
CMD_STEP = 1
CMD_CLOSE = 2

def worker_process(remote, parent_remote, plan, init_batches, use_implicit):
    """
    Läuft isoliert auf einem CPU-Kern.
    """
    parent_remote.close()

    try:
        # Eigene Simulation pro Prozess mit korrektem Modus starten
        env = SimpyShopWrapper(plan, init_batches, use_implicit_batch=use_implicit)

        while True:
            cmd, data = remote.recv()

            if cmd == CMD_STEP:
                state, reward, done, info = env.step(data)
                if done:
                    state = env.reset()
                remote.send((state, reward, done, info))

            elif cmd == CMD_RESET:
                state = env.reset()
                remote.send(state)

            elif cmd == CMD_CLOSE:
                remote.close()
                break
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Worker Error: {e}")
        traceback.print_exc() # Hilft beim Debuggen
    finally:
        remote.close()


class SubprocVecEnv:
    """
    Startet n_envs Prozesse und kommuniziert via Pipes.
    """

    def __init__(self, plan, init_batches, n_envs=4, use_implicit=False):
        self.n_envs = n_envs
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(n_envs)])
        self.ps = []

        for work_remote, remote in zip(self.work_remotes, self.remotes):
            # Arguments update: use_implicit übergeben
            p = mp.Process(target=worker_process, args=(work_remote, remote, plan, init_batches, use_implicit))
            p.daemon = True
            p.start()
            self.ps.append(p)

        for remote in self.work_remotes:
            remote.close()

    def reset(self):
        for remote in self.remotes:
            remote.send((CMD_RESET, None))
        return np.stack([remote.recv() for remote in self.remotes])

    def step(self, actions: list):
        for remote, action in zip(self.remotes, actions):
            remote.send((CMD_STEP, action))

        results = [remote.recv() for remote in self.remotes]
        states, rewards, dones, infos = zip(*results)
        return np.stack(states), np.array(rewards), np.array(dones), infos

    def close(self):
        for remote in self.remotes:
            remote.send((CMD_CLOSE, None))
        for p in self.ps:
            p.join()