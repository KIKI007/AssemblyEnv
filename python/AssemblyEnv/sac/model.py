import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
from torch_geometric.nn.models import GAT
from torch_geometric.nn import GATConv, Sequential
from torch.nn import Linear, ReLU, Tanh

def initialize_weights_he(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class BaseNetwork(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"))

    def set_graph(self, batch_size, n_part, edge_index, edge_attr):
        self.n_part = n_part
        n_edge = edge_index.shape[1]
        self.edge_index = edge_index
        self.edge_attr = edge_attr

        self.batch_edge_attr = edge_attr.repeat(batch_size, 1)
        self.batch_edge_index = edge_index.repeat(1, batch_size)
        for i in range(batch_size):
            self.batch_edge_index[:, n_edge * i: n_edge * (i + 1)] += torch.ones((2, n_edge), dtype=torch.long) * n_part * i;

    def batch_to_graph(self, batch):
        batch_0 = batch[:, :self.n_part].clone()
        batch_1 = batch[:, self.n_part: ].clone()
        graph_0 = batch_0.reshape(-1, 1)
        graph_1 = batch_1.reshape(-1, 1)
        return torch.cat([graph_0, graph_1], dim=1)

    def graph_to_batch(self, graph):
        graph_0 = graph[:, 0].clone()
        graph_1 = graph[:, 1].clone()
        batch_0 = graph_0.reshape(-1, self.n_part)
        batch_1 = graph_1.reshape(-1, self.n_part)
        return torch.cat([batch_0, batch_1], dim=1)

class QNetwork(BaseNetwork):
    def __init__(self, hidden_channels=16):

        super().__init__()

        self.head = Sequential('x, edge_index, edge_attr',[
            (GATConv(2, hidden_channels), 'x, edge_index, edge_attr -> x'),
            ReLU(inplace=True),
            (GATConv(hidden_channels, hidden_channels), 'x, edge_index, edge_attr -> x'),
            ReLU(inplace=True),
            (GATConv(hidden_channels, hidden_channels), 'x, edge_index, edge_attr -> x'),
            ReLU(inplace=True),
            (GATConv(hidden_channels, hidden_channels), 'x, edge_index, edge_attr -> x'),
            Linear(hidden_channels, 2),
            Tanh()]
        )

    def forward(self, states):
        node_feat = self.batch_to_graph(states)
        if states.shape[0] == 1:
            graphout = self.head(x = node_feat, edge_index = self.edge_index, edge_attr = self.edge_attr)
        else:
            graphout = self.head(x = node_feat, edge_index = self.batch_edge_index, edge_attr = self.batch_edge_attr)
        return self.graph_to_batch(graphout)

class TwinnedQNetwork(BaseNetwork):
    def __init__(self, hidden_channels=16):
        super().__init__()
        self.Q1 = QNetwork(hidden_channels)
        self.Q2 = QNetwork(hidden_channels)

    def forward(self, states):
        q1 = self.Q1(states)
        q2 = self.Q2(states)
        return q1, q2

    def set_graph(self, batch_size, n_part, edge_index, edge_attr):
        self.Q1.set_graph(batch_size, n_part, edge_index, edge_attr)
        self.Q2.set_graph(batch_size, n_part, edge_index, edge_attr)

class CateoricalPolicy(BaseNetwork):

    def __init__(self, hidden_channels=16):
        super().__init__()

        self.head = Sequential('x, edge_index, edge_attr', [
            (GATConv(2, hidden_channels), 'x, edge_index, edge_attr -> x'),
            ReLU(inplace=True),
            (GATConv(hidden_channels, hidden_channels), 'x, edge_index, edge_attr -> x'),
            ReLU(inplace=True),
            (GATConv(hidden_channels, hidden_channels), 'x, edge_index, edge_attr -> x'),
            ReLU(inplace=True),
            (GATConv(hidden_channels, hidden_channels), 'x, edge_index, edge_attr -> x'),
            Linear(hidden_channels, 2)]
        )

        self.inf = 1E8

    def mask_logits(self, states):
        node_feat = self.batch_to_graph(states)
        if states.shape[0] == 1:
            graph_action_logits = self.head(x = node_feat, edge_index = self.edge_index, edge_attr = self.edge_attr)
        else:
            graph_action_logits = self.head(x = node_feat, edge_index = self.batch_edge_index, edge_attr = self.batch_edge_attr)
        action_logits = self.graph_to_batch(graph_action_logits)

        install_states = states[:, : self.n_part]
        fixed_states = states[:, self.n_part :]

        used_robot = torch.sum(fixed_states, -1) >= 3
        robot_penalty = torch.einsum("i, j -> ij", used_robot, torch.ones(self.n_part, device=states.device))

        action_logits[:, : self.n_part] += robot_penalty * (-self.inf)
        action_logits[:, : self.n_part] += install_states * (-self.inf)
        action_logits[:, self.n_part :] += (1 - fixed_states) * (-self.inf)
        action_logits[:, -1] = -self.inf
        return action_logits

    def act(self, states):
        action_logits = self.mask_logits(states)
        greedy_actions = torch.argmax(
            action_logits, dim=1, keepdim=True)
        return greedy_actions

    def sample(self, states):
        action_logits = self.mask_logits(states)
        action_probs = F.softmax(action_logits, dim=1)
        action_dist = Categorical(action_probs)
        actions = action_dist.sample().view(-1, 1)

        # Avoid numerical instability.
        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs
