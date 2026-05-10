import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GraphNorm
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.data import Batch


class GNNFeatureExtractor(nn.Module):
    """
    Ein Graph Neural Network, das den Fabrik-Graphen in einen Vektor (Embedding) umwandelt.
    Architektur: 2-Layer GATv2 + Global Pooling.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, heads=1):
        super().__init__()

        # 1. Hop & GraphNorm
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=heads, concat=False)
        self.norm1 = GraphNorm(hidden_dim)

        # 2. Hop & GraphNorm
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False)
        self.norm2 = GraphNorm(hidden_dim)

        # 3. Hop & GraphNorm (Vergrößert das Sichtfeld)
        self.conv3 = GATv2Conv(hidden_dim, output_dim, heads=heads, concat=False)
        self.norm3 = GraphNorm(output_dim)

        # Global Attention Pooling (Öffnet den Flaschenhals)
        self.gate_nn = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.pool = AttentionalAggregation(gate_nn=self.gate_nn)

    def forward(self, batch_data):
        x, edge_index, batch = batch_data.x, batch_data.edge_index, batch_data.batch

        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.norm1(x, batch)
        x = torch.relu(x)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.norm2(x, batch)
        x = torch.relu(x)

        # Layer 3
        x = self.conv3(x, edge_index)
        x = self.norm3(x, batch)
        x = torch.relu(x)

        # Pooling mit Attention
        x = self.pool(x, batch)

        return x


if __name__ == "__main__":
    # Kurzer Test, ob das Modell kompiliert
    model = GNNFeatureExtractor(input_dim=6, hidden_dim=32, output_dim=64)
    print("GNN Modell erfolgreich erstellt:")
    print(model)