import pickle

import dgl

with open('./output/dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)
print(dataset)

import torch
from torch.utils.data import Dataset, DataLoader
import random
from model import MyModel
import torch.nn as nn

random.shuffle(dataset)
# train_size = int(0.8 * len(dataset))
# train_set = dataset[:train_size]
# test_set = dataset[train_size:]

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        g, edge_labels = self.data[idx]
        return g, edge_labels

# train_dataset = MyDataset(train_set)
# test_dataset = MyDataset(test_set)
# print(test_dataset[100])

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


from torch.utils.data.sampler import SubsetRandomSampler
from dgl.dataloading import GraphDataLoader

num_examples = len(dataset)
num_train = int(num_examples * 0.8)

train_indices = list(range(num_train))
test_indices = list(range(num_train, num_examples))

train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_dataset = MyDataset(dataset)
test_dataset = MyDataset(dataset)

train_dataloader = GraphDataLoader(
    train_dataset, batch_size=256, sampler=train_sampler, drop_last=False,
    collate_fn=lambda batch: (dgl.batch([item[0] for item in batch]), torch.stack([item[1] for item in batch]))
)
test_dataloader = GraphDataLoader(
    test_dataset, batch_size=256, sampler=test_sampler, drop_last=False,
    collate_fn=lambda batch: (dgl.batch([item[0] for item in batch]), torch.stack([item[1] for item in batch]))
)









#sageconv
num_classes = 5
hid_dim = 600
out_dim = 256
NUM_EPOCHS = 200

model = MyModel(hid_dim, out_dim, num_classes)
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(NUM_EPOCHS):
    model.train()
    for g, edge_labels in train_dataloader:
        g = g.to(device)
        edge_labels = edge_labels.to(device)

        optimizer.zero_grad()
        edge_scores= model(g, g.ndata['feature'])
        edge_scores = edge_scores.reshape(-1, num_classes)
        loss = criterion(edge_scores, edge_labels.view(-1))
        print(loss)
        loss.backward()
        optimizer.step()
        scheduler.step()

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for g, edge_labels in test_dataloader:
        g = g.to(device)
        edge_labels = edge_labels.to(device)

        edge_scores = model(g, g.ndata['feature'])
        edge_scores = edge_scores.reshape(-1, num_classes)
        _, predicted = torch.max(edge_scores.data, 1)
        total += edge_labels.size(0)*20
        correct += (predicted == edge_labels.view(-1)).sum().item()
        print(predicted)

    print(correct)
    print(total)
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')





# import gat_model
#
#
# #gat
# num_classes = 5
# hid_dim = 512
# out_dim = 256
# NUM_EPOCHS = 200
# num_heads = 5
#
# model = gat_model.MyModel_GAT(hid_dim, out_dim, num_classes,num_heads)
# criterion = nn.CrossEntropyLoss()
# # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
#
# for epoch in range(NUM_EPOCHS):
#     model.train()
#     for g, edge_labels in train_dataloader:
#         g = g.to(device)
#         edge_labels = edge_labels.to(device)
#
#         optimizer.zero_grad()
#         edge_scores= model(g, g.ndata['feature'])
#         edge_scores = edge_scores.reshape(-1, num_classes)
#         loss = criterion(edge_scores, edge_labels.view(-1))
#         print(loss)
#         loss.backward()
#         optimizer.step()
#         scheduler.step()
#
# model.eval()
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for g, edge_labels in test_dataloader:
#         g = g.to(device)
#         edge_labels = edge_labels.to(device)
#
#         edge_scores = model(g, g.ndata['feature'])
#         edge_scores = edge_scores.reshape(-1, num_classes)
#         _, predicted = torch.max(edge_scores.data, 1)
#         total += edge_labels.size(0)*20
#         correct += (predicted == edge_labels.view(-1)).sum().item()
#         print(predicted)
#
#     print(correct)
#     print(total)
#     accuracy = correct / total
#     print(f'Test Accuracy: {accuracy:.4f}')