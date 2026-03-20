
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, h, adj):
        h = torch.matmul(adj, h)
        return self.linear(h)

class EvolutionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mutation = nn.Linear(dim, dim)

    def forward(self, h):
        noise = torch.randn_like(h) * 0.01
        return h + self.mutation(h) * noise

class BiocentricGraphCNN(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.conv1 = GraphConv(hidden_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)

        self.evolution = EvolutionModule(hidden_dim)

        self.decoder = nn.Linear(hidden_dim, hidden_dim)
        self.policy = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h, adj, phi):
        h = F.relu(self.conv1(h, adj))
        h = F.relu(self.conv2(h, adj))

        # Evolutionary dynamics (life)
        h = self.evolution(h)

        reality = torch.tanh(self.decoder(h)) * phi

        # Free will (choice among actions)
        action_logits = self.policy(h)
        action = torch.softmax(action_logits, dim=-1)

        return h, reality, action
