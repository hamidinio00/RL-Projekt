import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from dataclasses import dataclass
from typing import Tuple, List

# Importiere dein GNN Modell
from learning_GNN.model_gnn import GNNFeatureExtractor


@dataclass
class PPOConfigGNN:
    # GNN Parameter
    node_feat_dim: int = 16  # Muss zum Wrapper passen
    gnn_hidden_dim: int = 64
    gnn_output_dim: int = 128  # Output des Message Passings
    gnn_heads: int = 1  # 1 reicht meist und ist schneller (wie besprochen)

    # MLP Parameter nach dem GNN (NEU: Mehr Denk-Kapazität)
    mlp_hidden_dims: Tuple[int, ...] = (256, 128)

    # Output Dimensionen
    max_prio_level: int = 6
    release_options: int = 5
    capacity_level_options: int = 6
    strategy_options: int = 4

    # NEU: Feine Granularität für den impliziten Modus
    implicit_weight_options: int = 25

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


class GNNAgent(nn.Module):
    def __init__(self, config: PPOConfigGNN):
        super().__init__()
        self.config = config
        self.use_implicit = config.use_implicit_batch

        # 1. GNN als "Auge" (Räumliche Wahrnehmung)
        self.gnn = GNNFeatureExtractor(
            input_dim=config.node_feat_dim,
            hidden_dim=config.gnn_hidden_dim,
            output_dim=config.gnn_output_dim,
            heads=config.gnn_heads
        )

        # 2. Shared MLP nach dem GNN (Die "Denk-Ebene")
        # Das GNN liefert einen Vektor der Größe gnn_output_dim.
        # Wir schicken diesen noch durch ein MLP, bevor die spezifischen Heads kommen.
        mlp_layers = []
        in_dim = config.gnn_output_dim
        for h_dim in config.mlp_hidden_dims:
            layer = nn.Linear(in_dim, h_dim)
            init_layer(layer)
            mlp_layers.append(layer)
            mlp_layers.append(nn.ReLU())
            in_dim = h_dim

        self.shared_mlp = nn.Sequential(*mlp_layers)
        last_dim = config.mlp_hidden_dims[-1]  # Output Dimension des MLPs

        # 3. Actor Heads (Die "Hände" - Entscheidungen treffen)
        self.head_disassembly = self._make_head(last_dim, config.max_prio_level)
        self.head_inspection = self._make_head(last_dim, config.max_prio_level)
        self.head_cleaning = self._make_head(last_dim, config.max_prio_level)
        self.head_repair = self._make_head(last_dim, config.max_prio_level)
        self.head_assembly = self._make_head(last_dim, config.max_prio_level)

        self.head_release = self._make_head(last_dim, config.release_options)
        self.head_batch_size = self._make_head(last_dim, len(config.batch_sizes))
        self.head_capacity = self._make_head(last_dim, config.capacity_level_options)

        # Conditional Heads (Implizit vs Explizit)
        if self.use_implicit:
            # Shared Mixer Layer für Gewichte (Wie im flachen Agenten)
            self.implicit_mixer = nn.Sequential(
                nn.Linear(last_dim, 64),
                nn.ReLU()
            )
            # Höhere Auflösung (z.B. 0-24)
            self.head_prio_low = self._make_head(64, config.implicit_weight_options)
            self.head_prio_mid = self._make_head(64, config.implicit_weight_options)
            self.head_prio_high = self._make_head(64, config.implicit_weight_options)
        else:
            self.head_strategy = self._make_head(last_dim, config.strategy_options)

        # 4. Critic Head (Wertschätzer)
        self.critic_head = nn.Linear(last_dim, 1)
        init_layer(self.critic_head, std=1.0)

    def _make_head(self, in_dim, out_dim):
        l = nn.Linear(in_dim, out_dim)
        init_layer(l, std=0.01)
        return l

    def forward(self, data, deterministic: bool = False):
        """
        data: PyTorch Geometric Batch Objekt (Graph)
        """
        # 1. Graph durch GNN schicken -> Rohes Graph-Embedding
        graph_embed = self.gnn(data)

        # 2. Durch Shared MLP schicken -> Verarbeitete Features
        features = self.shared_mlp(graph_embed)

        # 3. Value berechnen
        value = self.critic_head(features)

        # 4. Aktionen berechnen
        actions = {}

        # Helper zum Abrufen der Action
        def get_action(head, feat, key):
            logits = head(feat)
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
            mix_feat = self.implicit_mixer(features)
            actions['batch_prio_low'] = get_action(self.head_prio_low, mix_feat, 'batch_prio_low')
            actions['batch_prio_mid'] = get_action(self.head_prio_mid, mix_feat, 'batch_prio_mid')
            actions['batch_prio_high'] = get_action(self.head_prio_high, mix_feat, 'batch_prio_high')
        else:
            actions['batch_strategy'] = get_action(self.head_strategy, features, 'batch_strategy')

        return actions, value

    def evaluate(self, data, actions):
        """
        Wird für den PPO Update Schritt (Backward Pass) benötigt.
        """
        graph_embed = self.gnn(data)
        features = self.shared_mlp(graph_embed)
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