#************************data preprocessing***************************


# import pandas as pd
# import json
# data = pd.read_pickle("./output/data.pkl")
# # print(data)
# for i in data:
# #     for k in i['edges']:
# #         print(k[:2])
#     print(i['types'])

#create dataset

# import pickle
# import numpy as np
# import dgl
# import torch
# from sklearn.preprocessing import OneHotEncoder
#
# enc = OneHotEncoder(sparse=False)
#
# with open('./output/data.pkl', 'rb') as f:
#     data_list = pickle.load(f)
#
# graphs = []
# node_features = []
# labels = []
#
# for item in data_list:
#     node_types = item['types']
#     one_hot_encoded = enc.fit_transform(np.expand_dims(node_types, axis=1)).tolist()
#     node_features.append(torch.tensor(one_hot_encoded).float())
#
#     src = np.array(item['edges'])[:, 0]
#     dst = np.array(item['edges'])[:, 1]
#     edge_labels = np.array(item['edges'])[:, 2]
#
#     all_src = np.concatenate([src])
#     all_dst = np.concatenate([dst])
#
#     g = dgl.DGLGraph((all_src, all_dst))
#     g.ndata['feature'] = node_features[-1]
#
#     g.edata['label'] = torch.tensor(edge_labels)
#
#     graphs.append(g)
#     labels.append(edge_labels)

#
# for i, (g, nf, l) in enumerate(zip(graphs, node_features, labels)):
#     g_name = f'graph_{i}.dgl'
#     dgl.save_graphs(g_name, [g])
#
#     feature_name = f'node_feature_{i}.pt'
#     torch.save(nf, feature_name)
#
#     label_name = f'label_{i}.npy'
#     np.save(label_name, l)
import pandas as pd
import pickle
import numpy as np
import dgl
import torch
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# node_categories = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12,13,14,15,16,17,18]

# ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(sparse=False), [0])], remainder='passthrough')
# ct.fit(np.array([[i] for i in node_categories]))

with open('./output/data_6.pkl', 'rb') as f:
    data_list = pickle.load(f)

all_node_features = []
all_edge_labels = []
graph_files = []
label_files = []
dataset = []
count = 0

for item_idx, item in enumerate(data_list):
    node_types = item['types']
    # node_types_df = pd.DataFrame({'node_type': node_types})
    # one_hot_encoded = ct.transform(node_types_df)
    # node_features = torch.tensor(one_hot_encoded).float()
    excess_node = []
    for idnode in range(len(node_types)):
        if node_types[idnode] == 10 or 12<=node_types[idnode]<=17 :
            excess_node.append(idnode)

    old = node_types
    new_node_types = [k for k in node_types if k !=10 and not (12<=k<=17)]

    edges = item['edges']
    filtered_edges = [(src, dst, lbl) for src, dst, lbl in edges if lbl in [3, 7, 6, 2] and src not in excess_node and dst not in excess_node]
    src = [e[0] for e in filtered_edges]
    dst = [e[1] for e in filtered_edges]
    # print(src)
    # print(dst)
    # print(old)
    for k in src:
        k = new_node_types.index(old[k])
    for k in dst:
        k = new_node_types.index(old[k])
    edge_labels = [e[2] for e in filtered_edges]
    for k in range(len(edge_labels)):
        if edge_labels[k] == 3:
            edge_labels[k] = 0
        elif edge_labels[k]==7:
            edge_labels[k]=1
        elif edge_labels[k]==6:
            edge_labels[k]=2
        elif edge_labels[k]==2:
            edge_labels[k]=3
    # print(edge_labels)
    src_1 = src
    dst_1 = dst
    # print(src_1)
    # print(dst_1)
    # print(edge_labels)

    for k in range(15-len(edge_labels)):
        edge_labels.append(4)              # 4 self
        src_1.append(0)
        dst_1.append(0)
    # if len(edge_labels)==0:
    #     continue
    #
    # acount = int(15/len(edge_labels))-1
    # bcount = 15%len(edge_labels)
    # for i in range(acount):
    #     for k in range(len(edge_labels)):
    #         edge_labels.append(edge_labels[k])
    #         src_1.append(src_1[k])
    #         dst_1.append(dst_1[k])
    #
    # for i in range(bcount):
    #     edge_labels.append(edge_labels[i])
    #     src_1.append(src_1[i])
    #     dst_1.append(dst_1[i])

    edge_labels = np.array(edge_labels)
    edge_labels = torch.tensor(edge_labels)

    src_1 = src_1[:15]
    dst_1 = dst_1[:15]
    edge_labels = edge_labels[:15]
    print(src_1)
    print(dst_1)
    # print(new_node_types)
    # print(edge_labels)
    src_1 = np.array(src_1)
    dst_1 = np.array(dst_1)

    all_src = np.concatenate([src_1])
    all_dst = np.concatenate([dst_1])
    g = dgl.DGLGraph((all_src, all_dst))
    if len(g.nodes()) != len(new_node_types):
        print('err')
        continue
    count = count+1

    g.ndata['feature'] = torch.ones(g.num_nodes(),8)
    # print(g.ndata['feature'])
    for k in range(len(new_node_types)):
        if new_node_types[k] == 0:
            g.ndata['feature'][k] =  torch.tensor([1.,0.,0.,0.,0.,0.,0.,0.])
        elif new_node_types[k] == 1:
            g.ndata['feature'][k] =  torch.tensor([0.,1.,0.,0.,0.,0.,0.,0.])
        elif new_node_types[k] == 2:
            g.ndata['feature'][k] =  torch.tensor([0.,0.,1.,0.,0.,0.,0.,0.])
        elif new_node_types[k] == 3:
            g.ndata['feature'][k] =  torch.tensor([0.,0.,0.,1.,0.,0.,0.,0.])
        elif new_node_types[k] == 4:
            g.ndata['feature'][k] =  torch.tensor([0.,0.,0.,0.,1.,0.,0.,0.])
        elif new_node_types[k] == 5:
            g.ndata['feature'][k] =  torch.tensor([0.,1.,0.,0.,0.,0.,0.,0.])
        elif new_node_types[k] == 6:
            g.ndata['feature'][k] =  torch.tensor([0.,0.,0.,0.,0.,1.,0.,0.])
        elif new_node_types[k] == 7:
            g.ndata['feature'][k] =  torch.tensor([0.,1.,0.,0.,0.,0.,0.,0.])
        elif new_node_types[k] == 8:
            g.ndata['feature'][k] =  torch.tensor([0.,1.,0.,0.,0.,0.,0.,0.])
        elif new_node_types[k] == 9:
            g.ndata['feature'][k] =  torch.tensor([0.,0.,0.,0.,0.,0.,1.,0.])
        elif new_node_types[k] == 11:
            g.ndata['feature'][k] =  torch.tensor([0.,0.,0.,0.,0.,0.,0.,1.])

    # print(g.nodes)
    # print(g.ndata['feature'])
    dataset.append((g, edge_labels))

print(count)
with open('./output/dataset_6_add0.pkl', 'wb') as f:
    pickle.dump(dataset, f)

# print(dataset)

