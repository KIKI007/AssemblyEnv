import torch

from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from torch_geometric.nn import ChebConv, GCNConv, MessagePassing
from torch_geometric.nn.models import GCN, GraphSAGE, GAT, EdgeCNN
from AssemblyEnv.py_rigidblock import Assembly
from AssemblyEnv.geometry import AssemblyCheckerMosek
class MyConv(MessagePassing):
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

def test_static_graph():
    torch.manual_seed(10)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    x1, x2 = torch.randn(3, 8), torch.randn(3, 8)
    e1 = torch.randn(int(edge_index.shape[1]), 8)
    e2 = torch.randn(int(edge_index.shape[1]), 8)
    print(e1)

    data1 = Data(edge_index=edge_index, x=x1)
    data2 = Data(edge_index=edge_index, x=x2)

    x = torch.stack([x1, x2], dim=0)
    ew = torch.stack([e1, e2], dim=0)

    for conv in [EdgeCNN(8, 16, 3, 32)]:
        out2 = conv.forward(x1, edge_index, edge_attr=e1)
        print(out2.shape)


def test_part_graph():
    parts = [[[4.0, 3.0], [5.0, 3.0], [5.0, 5.0], [4.0, 5.0]],
             [[0.0, 3.0], [1.0, 3.0], [1.0, 5.0], [0.0, 5.0]],
             [[2.0, 3.0], [3.0, 3.0], [3.0, 5.0], [2.0, 5.0]],
             [[2.0, 0.0], [3.0, 0.0], [3.0, 2.0], [2.0, 2.0]],
             [[0.0, 0.0], [1.0, 0.0], [1.0, 2.0], [0.0, 2.0]],
             [[4.0, 0.0], [5.0, 0.0], [5.0, 2.0], [4.0, 2.0]],
             [[0.0, 2.0], [5.0, 2.0], [5.0, 3.0], [0.0, 3.0]],
             [[0.0, 5.0], [5.0, 5.0], [5.0, 6.0], [0.0, 6.0]]]

    assembly = AssemblyCheckerMosek(parts).assembly
    partList = [x for x in range(assembly.n_part())]
    contacts = assembly.contacts(partList, 1.0)
    analyzer = assembly.analyzerGNN(contacts, False)
    [e0, e1, edge_attr] = analyzer.gnn()
    e0 = torch.tensor(e0, dtype = torch.long)
    e1 = torch.tensor(e1, dtype = torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    edge_index = torch.vstack([e0, e1])

    n = assembly.n_part()
    x1 = torch.randn(n, 2)
    x2 = torch.randn(n, 2)

    data1 = Data(num_nodes = n, edge_index=edge_index, x=x1, edge_attr=edge_attr)
    data2 = Data(num_nodes = n, edge_index=edge_index, x=x2, edge_attr=edge_attr)

    data_list = [data1, data2]
    loader = DataLoader(data_list, batch_size=2)
    batch = next(iter(loader))
    nB = len(data_list)
    outchannel = 1
    model = GAT(2, 16, 3, outchannel)
    out = model.forward(batch.x, batch.edge_index, edge_attr=batch.edge_attr)
    print(out)
    out1 = out.reshape(nB, n)
    print(out1)
    print(torch.sum(out1, dim=-1))



test_part_graph()