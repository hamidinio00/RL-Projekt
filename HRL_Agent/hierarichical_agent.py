import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical


def init_layer(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ManagerNetwork(nn.Module):
    """
    Obere Ebene: Entscheidet über Kapazität, Batch-Logik und Auftragsfreigabe.
    UPGRADE: Tiefere Layer (512, 256, 128), Implicit Mixer und 21 Optionen.
    """

    def __init__(self, input_dim, hidden_dims=(512, 256, 128), use_implicit=False, implicit_weight_options=21):
        super().__init__()
        self.use_implicit = use_implicit
        self.implicit_weight_options = implicit_weight_options

        # Shared Feature Extractor (Body)
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            l = nn.Linear(in_dim, h_dim)
            init_layer(l)
            layers.append(l)
            layers.append(nn.ReLU())  # Tanh für Stabilität, ReLu für bessere Ergebnisse
            in_dim = h_dim
        self.body = nn.Sequential(*layers)

        last_dim = hidden_dims[-1]

        # 1. Basis-Köpfe
        self.head_capacity = self._make_head(last_dim, 6)
        self.head_release = self._make_head(last_dim, 5)
        self.head_batch = self._make_head(last_dim, 11)

        # 2. Modus-abhängige Köpfe
        if self.use_implicit:
            # UPGRADE: Implicit Mixer, damit die Köpfe korrelieren
            self.implicit_mixer = nn.Sequential(
                nn.Linear(last_dim, 64),
                nn.ReLU()
            )
            self.head_prio_low = self._make_head(64, self.implicit_weight_options)
            self.head_prio_mid = self._make_head(64, self.implicit_weight_options)
            self.head_prio_high = self._make_head(64, self.implicit_weight_options)
        else:
            self.head_strategy = self._make_head(last_dim, 4)

        # Critic
        self.critic = nn.Linear(last_dim, 1)
        init_layer(self.critic, std=1.0)

    def _make_head(self, in_dim, out_dim):
        l = nn.Linear(in_dim, out_dim)
        init_layer(l, std=0.01)
        return l

    def forward(self, state):
        x = self.body(state)

        dists = {
            'capacity': Categorical(logits=self.head_capacity(x)),
            'release': Categorical(logits=self.head_release(x)),
            'batch': Categorical(logits=self.head_batch(x))
        }

        if self.use_implicit:
            mix_feat = self.implicit_mixer(x)
            dists['prio_low'] = Categorical(logits=self.head_prio_low(mix_feat))
            dists['prio_mid'] = Categorical(logits=self.head_prio_mid(mix_feat))
            dists['prio_high'] = Categorical(logits=self.head_prio_high(mix_feat))
        else:
            dists['strategy'] = Categorical(logits=self.head_strategy(x))

        value = self.critic(x)
        return dists, value


class WorkerNetwork(nn.Module):
    """
    Untere Ebene: Entscheidet über Mitarbeiterverteilung.
    UPGRADE: Tiefere Layer (512, 256, 128) und korrekt skalierte Embeddings!
    """

    def __init__(self, state_dim, context_dim=6, hidden_dims=(512, 256, 128), implicit_weight_options=21):
        super().__init__()
        self.context_dim = context_dim
        emb_size = 4

        # Basis Embeddings
        self.emb_cap = nn.Embedding(6, emb_size)
        self.emb_rel = nn.Embedding(5, emb_size)
        self.emb_batch = nn.Embedding(11, emb_size)

        # Modus Embeddings
        if self.context_dim == 6:
            # WICHTIG: Die Embedding-Größe muss an die Manager-Ausgabe (21 Optionen) angepasst sein!
            self.emb_prio_low = nn.Embedding(implicit_weight_options, emb_size)
            self.emb_prio_mid = nn.Embedding(implicit_weight_options, emb_size)
            self.emb_prio_high = nn.Embedding(implicit_weight_options, emb_size)
        else:
            self.emb_strat = nn.Embedding(4, emb_size)

        total_input_dim = state_dim + (context_dim * emb_size)

        layers = []
        in_dim = total_input_dim
        for h_dim in hidden_dims:
            l = nn.Linear(in_dim, h_dim)
            init_layer(l)
            layers.append(l)
            layers.append(nn.ReLU())  # anh für Stabilität, ReLu für bessere Ergebnisse
            in_dim = h_dim
        self.body = nn.Sequential(*layers)

        # Heads (Worker Priorities 0..5 bleiben unverändert)
        self.head_prio_insp = self._make_head(in_dim, 6)
        self.head_prio_dis = self._make_head(in_dim, 6)
        self.head_prio_clean = self._make_head(in_dim, 6)
        self.head_prio_rep = self._make_head(in_dim, 6)
        self.head_prio_ass = self._make_head(in_dim, 6)

        self.critic = nn.Linear(in_dim, 1)
        init_layer(self.critic, std=1.0)

    def _make_head(self, in_dim, out_dim):
        l = nn.Linear(in_dim, out_dim)
        init_layer(l, std=0.01)
        return l

    def forward(self, state, ctx_dict):
        e_cap = self.emb_cap(ctx_dict['capacity'])
        e_rel = self.emb_rel(ctx_dict['release'])
        e_batch = self.emb_batch(ctx_dict['batch'])

        emb_list = [e_cap, e_rel, e_batch]

        if self.context_dim == 6:
            e_pl = self.emb_prio_low(ctx_dict['prio_low'])
            e_pm = self.emb_prio_mid(ctx_dict['prio_mid'])
            e_ph = self.emb_prio_high(ctx_dict['prio_high'])
            emb_list.extend([e_pl, e_pm, e_ph])
        else:
            e_strat = self.emb_strat(ctx_dict['strategy'])
            emb_list.append(e_strat)

        context_vec = torch.cat(emb_list, dim=-1)

        if state.dim() == 1:
            state = state.unsqueeze(0)

        x = torch.cat([state, context_vec], dim=-1)
        features = self.body(x)

        return {
            'prio_insp': Categorical(logits=self.head_prio_insp(features)),
            'prio_dis': Categorical(logits=self.head_prio_dis(features)),
            'prio_clean': Categorical(logits=self.head_prio_clean(features)),
            'prio_rep': Categorical(logits=self.head_prio_rep(features)),
            'prio_ass': Categorical(logits=self.head_prio_ass(features)),
        }, self.critic(features)