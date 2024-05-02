import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATLayer, self).__init__()
        # self.g = g
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g,h):
        # equation (1)
        z = self.fc(h)
        g.ndata['z'] = z
        # equation (2)
        g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop('h')


class MultiHeadGATLayer(nn.Module):
    def __init__(self,  in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer( in_dim, out_dim))
        self.merge = merge

    def forward(self, g,h):
        head_outs = [attn_head(g,h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))


class GAT(nn.Module):
    def __init__(self,  in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer( in_dim, hidden_dim, num_heads)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = MultiHeadGATLayer( hidden_dim * num_heads, hidden_dim, num_heads)
        self.layer3 = MultiHeadGATLayer( hidden_dim * num_heads, hidden_dim, num_heads)
        self.layer4 = MultiHeadGATLayer( hidden_dim * num_heads, hidden_dim, num_heads)
        self.layer5 = MultiHeadGATLayer( hidden_dim * num_heads, hidden_dim, num_heads)
        self.layer6 = MultiHeadGATLayer( hidden_dim * num_heads, hidden_dim, num_heads)
        self.layer7 = MultiHeadGATLayer( hidden_dim * num_heads, hidden_dim, num_heads)
        self.layer8 = MultiHeadGATLayer( hidden_dim * num_heads, hidden_dim, num_heads)
        self.layer9 = MultiHeadGATLayer( hidden_dim * num_heads, hidden_dim, num_heads)
        self.layer10 = MultiHeadGATLayer( hidden_dim * num_heads, hidden_dim, num_heads)
        self.layer11 = MultiHeadGATLayer( hidden_dim * num_heads,  hidden_dim, num_heads)
        self.layer12 = MultiHeadGATLayer( hidden_dim * num_heads, hidden_dim, num_heads)
        self.layer13 = MultiHeadGATLayer( hidden_dim * num_heads, hidden_dim, num_heads)
        self.layer14 = MultiHeadGATLayer( hidden_dim * num_heads, hidden_dim, num_heads)
        self.layer15 = MultiHeadGATLayer( hidden_dim * num_heads, out_dim, 1)

    def forward(self, g,h):
        h = self.layer1(g,h)
        h = F.elu(h)
        h = self.layer2(g,h)
        h = F.elu(h)
        h = self.layer3(g,h)
        h = F.elu(h)
        h = self.layer4(g,h)
        h = F.elu(h)
        h = self.layer5(g,h)
        h = F.elu(h)
        h = self.layer6(g,h)
        h = F.elu(h)
        h = self.layer7(g,h)
        h = F.elu(h)
        h = self.layer8(g,h)
        h = F.elu(h)
        h = self.layer9(g,h)
        h = F.elu(h)
        h = self.layer10(g,h)
        h = F.elu(h)
        h = self.layer11(g,h)
        h = F.elu(h)
        h = self.layer12(g,h)
        h = F.elu(h)
        h = self.layer13(g,h)
        h = F.elu(h)
        h = self.layer14(g,h)
        h = F.elu(h)
        h = self.layer15(g,h)

        return h


class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = self.W(torch.cat([h_u, h_v], 1))
        return {'score': score}

    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']

class MyModel_GAT(nn.Module):
    """主模型：结构比较清晰"""
    def __init__(self, hid_dim, out_dim, num_classes,num_heads):
        super().__init__()
        # self.node_emb = nn.Embedding(num_nodes, emb_dim)
        self.gcn = GAT(19, hid_dim, out_dim,num_heads)
        self.edge_predictor = MLPPredictor(out_dim, num_classes)
        # self.node_predictor = nn.Sequential(
        #     nn.Linear(out_dim, hid_dim),
        #     nn.ReLU(),
        #     nn.Linear(hid_dim, 10)
        # )

    def forward(self, graph, in_features):
        # x = self.node_emb(input_nodes)
        x = self.gcn( graph,in_features)
        edge_scores = self.edge_predictor(graph, x)
        # node_scores = self.node_predictor(x)
        return edge_scores
