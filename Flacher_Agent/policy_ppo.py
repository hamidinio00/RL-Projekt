import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from typing import Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class PPOConfig:
    # INPUT DIM muss vom Wrapper gesetzt werden!
    input_dim: int = 64

    # OPTIMIERUNG 1: Größeres, tieferes Netzwerk
    hidden_dims: Tuple[int, ...] = (256, 126, 64)

    # Output Dimensionen
    max_prio_level: int = 6  # Für Maschinen (0-5 reicht)
    implicit_weight_options: int = 25
    release_options: int = 5
    capacity_level_options: int = 6
    strategy_options: int = 4

    batch_sizes: List[int] = None
    use_implicit_batch: bool = False

    # Hyperparameter
    lr_actor: float = 1e-4
    lr_critic: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    max_grad_norm: float = 0.5

    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30]


def init_layer(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PolicyNetwork(nn.Module):
    def __init__(self, config: PPOConfig):
        super().__init__()
        self.config = config

        # --- 1. Shared Body (Feature Extractor) ---
        layers = []
        in_dim = config.input_dim

        for h_dim in config.hidden_dims:
            layer = nn.Linear(in_dim, h_dim)
            init_layer(layer)
            layers.append(layer)
            layers.append(nn.ReLU())
            in_dim = h_dim

        self.shared_net = nn.Sequential(*layers)
        last_dim = config.hidden_dims[-1]

        # --- 2. Actor Heads ---
        # A. Standard Heads (Unabhängig)
        self.head_disassembly = self._make_head(last_dim, config.max_prio_level)
        self.head_inspection = self._make_head(last_dim, config.max_prio_level)
        self.head_cleaning = self._make_head(last_dim, config.max_prio_level)
        self.head_repair = self._make_head(last_dim, config.max_prio_level)
        self.head_assembly = self._make_head(last_dim, config.max_prio_level)

        self.head_release = self._make_head(last_dim, config.release_options)
        self.head_batch_size = self._make_head(last_dim, len(config.batch_sizes))
        self.head_capacity = self._make_head(last_dim, config.capacity_level_options)

        # B. Special Heads (Strategy vs Implicit)
        self.use_implicit = config.use_implicit_batch

        if self.use_implicit:
            # Shared Mixer Layer für Gewichte
            self.implicit_mixer = nn.Sequential(
                nn.Linear(last_dim, 64),
                nn.ReLU()
            )
            # Höhere Auflösung (0-20)
            self.head_prio_low = self._make_head(64, config.implicit_weight_options)
            self.head_prio_mid = self._make_head(64, config.implicit_weight_options)
            self.head_prio_high = self._make_head(64, config.implicit_weight_options)
        else:
            self.head_strategy = self._make_head(last_dim, config.strategy_options)

        # --- 3. Critic Head ---
        self.critic_head = nn.Linear(last_dim, 1)
        init_layer(self.critic_head, std=1.0)

    def _make_head(self, in_dim, out_dim):
        # Init mit kleinem std=0.01 sorgt für viel Zufall am Anfang (Exploration)
        l = nn.Linear(in_dim, out_dim)
        init_layer(l, std=0.01)
        return l

    def forward(self, state_tensor: torch.Tensor, deterministic: bool = False):
        # 1. Features extrahieren
        features = self.shared_net(state_tensor)

        # 2. Value berechnen
        value = self.critic_head(features)

        # 3. Actions berechnen
        actions = {}

        # Helper zum Abrufen der Action (Argmax oder Sample)
        def get_action(head, feat, key):
            logits = head(feat)
            # Clipping gegen NaN Fehler
            logits = torch.clamp(logits, -20, 20)

            if deterministic:
                return torch.argmax(logits, dim=-1)
            else:
                return Categorical(logits=logits).sample()

        # Standard Actions
        actions['prio_disassembly'] = get_action(self.head_disassembly, features, 'prio_disassembly')
        actions['prio_inspection'] = get_action(self.head_inspection, features, 'prio_inspection')
        actions['prio_cleaning'] = get_action(self.head_cleaning, features, 'prio_cleaning')
        actions['prio_repair'] = get_action(self.head_repair, features, 'prio_repair')
        actions['prio_assembly'] = get_action(self.head_assembly, features, 'prio_assembly')

        actions['order_release'] = get_action(self.head_release, features, 'order_release')
        actions['batch_size'] = get_action(self.head_batch_size, features, 'batch_size')
        actions['capacity_level'] = get_action(self.head_capacity, features, 'capacity_level')

        # Conditional Actions
        if self.use_implicit:
            # Features durch Mixer jagen
            mix_feat = self.implicit_mixer(features)
            actions['batch_prio_low'] = get_action(self.head_prio_low, mix_feat, 'batch_prio_low')
            actions['batch_prio_mid'] = get_action(self.head_prio_mid, mix_feat, 'batch_prio_mid')
            actions['batch_prio_high'] = get_action(self.head_prio_high, mix_feat, 'batch_prio_high')
        else:
            actions['batch_strategy'] = get_action(self.head_strategy, features, 'batch_strategy')

        return actions, value

    def evaluate(self, state_tensor, actions):
        """
        Wird für den PPO Update Schritt (Backward Pass) benötigt.
        Berechnet Log-Probs und Entropie.
        """
        features = self.shared_net(state_tensor)
        value = self.critic_head(features)

        total_log_prob = 0
        entropy = 0

        # Helper
        def eval_head(head, feat, action_val):
            logits = head(feat)
            logits = torch.clamp(logits, -20, 20)
            dist = Categorical(logits=logits)
            return dist.log_prob(action_val), dist.entropy()

        # Standard
        lp, ent = eval_head(self.head_disassembly, features, actions['prio_disassembly'])
        total_log_prob += lp
        entropy += ent

        lp, ent = eval_head(self.head_inspection, features, actions['prio_inspection'])
        total_log_prob += lp
        entropy += ent

        lp, ent = eval_head(self.head_cleaning, features, actions['prio_cleaning'])
        total_log_prob += lp
        entropy += ent

        lp, ent = eval_head(self.head_repair, features, actions['prio_repair'])
        total_log_prob += lp
        entropy += ent

        lp, ent = eval_head(self.head_assembly, features, actions['prio_assembly'])
        total_log_prob += lp
        entropy += ent

        lp, ent = eval_head(self.head_release, features, actions['order_release'])
        total_log_prob += lp
        entropy += ent

        lp, ent = eval_head(self.head_batch_size, features, actions['batch_size'])
        total_log_prob += lp
        entropy += ent

        lp, ent = eval_head(self.head_capacity, features, actions['capacity_level'])
        total_log_prob += lp
        entropy += ent

        # Conditional
        if self.use_implicit:
            mix_feat = self.implicit_mixer(features)

            lp, ent = eval_head(self.head_prio_low, mix_feat, actions['batch_prio_low'])
            total_log_prob += lp
            entropy += ent

            lp, ent = eval_head(self.head_prio_mid, mix_feat, actions['batch_prio_mid'])
            total_log_prob += lp
            entropy += ent

            lp, ent = eval_head(self.head_prio_high, mix_feat, actions['batch_prio_high'])
            total_log_prob += lp
            entropy += ent
        else:
            lp, ent = eval_head(self.head_strategy, features, actions['batch_strategy'])
            total_log_prob += lp
            entropy += ent

        return total_log_prob, value, entropy