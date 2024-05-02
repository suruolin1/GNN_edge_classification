import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl.function as fn
import dgl.nn as dglnn

import dgl
import numpy as np

class TwoLayerGCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(in_feats=in_dim, out_feats=hid_dim, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(in_feats=hid_dim, out_feats=hid_dim, aggregator_type='mean')
        self.conv3 = dglnn.SAGEConv(in_feats=hid_dim, out_feats=hid_dim, aggregator_type='mean')
        self.conv4 = dglnn.SAGEConv(in_feats=hid_dim, out_feats=hid_dim, aggregator_type='mean')
        self.conv5 = dglnn.SAGEConv(in_feats=hid_dim, out_feats=hid_dim, aggregator_type='mean')
        self.conv6 = dglnn.SAGEConv(in_feats=hid_dim, out_feats=hid_dim, aggregator_type='mean')
        self.conv7 = dglnn.SAGEConv(in_feats=hid_dim, out_feats=hid_dim, aggregator_type='mean')
        # self.conv8 = dglnn.SAGEConv(in_feats=hid_dim, out_feats=hid_dim, aggregator_type='mean')
        # self.conv9 = dglnn.SAGEConv(in_feats=hid_dim, out_feats=hid_dim, aggregator_type='mean')
        # self.conv10 = dglnn.SAGEConv(in_feats=hid_dim, out_feats=hid_dim, aggregator_type='mean')
        # self.conv11 = dglnn.SAGEConv(in_feats=hid_dim, out_feats=hid_dim, aggregator_type='mean')
        # self.conv12 = dglnn.SAGEConv(in_feats=hid_dim, out_feats=hid_dim, aggregator_type='mean')
        # self.conv13 = dglnn.SAGEConv(in_feats=hid_dim, out_feats=hid_dim, aggregator_type='mean')
        # self.conv14 = dglnn.SAGEConv(in_feats=hid_dim, out_feats=hid_dim, aggregator_type='mean')
        # self.conv15 = dglnn.SAGEConv(in_feats=hid_dim, out_feats=hid_dim, aggregator_type='mean')
        self.conv16 = dglnn.SAGEConv(in_feats=hid_dim, out_feats=out_dim, aggregator_type='mean')

        # self.conv1 = dglnn.GraphConv(in_dim, hid_dim, allow_zero_in_degree=True)
        # self.conv2 = dglnn.GraphConv(hid_dim, hid_dim, allow_zero_in_degree=True)
        # self.conv3 = dglnn.GraphConv(hid_dim, hid_dim, allow_zero_in_degree=True)
        # self.conv4 = dglnn.GraphConv(hid_dim, hid_dim, allow_zero_in_degree=True)
        # self.conv5 = dglnn.GraphConv(hid_dim, hid_dim, allow_zero_in_degree=True)
        # self.conv6 = dglnn.GraphConv(hid_dim, hid_dim, allow_zero_in_degree=True)
        # self.conv7 = dglnn.GraphConv(hid_dim, hid_dim, allow_zero_in_degree=True)
        # self.conv8 = dglnn.GraphConv(hid_dim, hid_dim, allow_zero_in_degree=True)
        # self.conv9 = dglnn.GraphConv(hid_dim, out_dim, allow_zero_in_degree=True)
    def forward(self, graph, x):
        x = F.relu(self.conv1(graph, x))
        x = F.relu(self.conv2(graph, x))
        x = F.relu(self.conv3(graph, x))
        x = F.relu(self.conv4(graph, x))
        x = F.relu(self.conv5(graph, x))
        x = F.relu(self.conv6(graph, x))
        x = F.relu(self.conv7(graph, x))
        # x = F.relu(self.conv8(graph, x))
        # x = F.relu(self.conv9(graph, x))
        # x = F.relu(self.conv10(graph, x))
        # x = F.relu(self.conv11(graph, x))
        # x = F.relu(self.conv12(graph, x))
        # x = F.relu(self.conv13(graph, x))
        # x = F.relu(self.conv14(graph, x))
        # x = F.relu(self.conv15(graph, x))
        x = F.relu(self.conv16(graph, x))

        return x

class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        # self.W = nn.Linear(in_features * 2, out_classes)
        self.mlp = nn.Sequential(
            nn.Linear(in_features * 2, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, out_classes)
        )


    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = self.mlp(torch.cat([h_u, h_v], 1))
        # score = self.mlp(torch.tensor((h_v*h_u).sum(dim = -1)))
        return {'score': score}

    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']

class MyModel(nn.Module):
    """主模型：结构比较清晰"""
    def __init__(self, hid_dim, out_dim, num_classes):
        super().__init__()
        # self.node_emb = nn.Embedding(num_nodes, emb_dim)
        self.gcn = TwoLayerGCN(8, hid_dim, out_dim)
        self.edge_predictor = MLPPredictor(out_dim, num_classes)
        # self.node_predictor = nn.Sequential(
        #     nn.Linear(out_dim, hid_dim),
        #     nn.ReLU(),
        #     nn.Linear(hid_dim, 10)
        # )

    def forward(self, graph, in_features):
        # x = self.node_emb(input_nodes)
        x = self.gcn(graph, in_features)
        edge_scores = self.edge_predictor(graph, x)
        # node_scores = self.node_predictor(x)
        return edge_scores
