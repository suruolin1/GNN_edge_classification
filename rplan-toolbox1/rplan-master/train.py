import pickle

import dgl
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.sgd import SGD

with open('./output/dataset_6_add0.pkl', 'rb') as f:
    dataset = pickle.load(f)
print(dataset)

with open('log1.txt', 'w') as file:
    pass

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
    train_dataset, batch_size=64, sampler=train_sampler, drop_last=False,
    collate_fn=lambda batch: (dgl.batch([item[0] for item in batch]), torch.stack([item[1] for item in batch]))
)
test_dataloader = GraphDataLoader(
    test_dataset, batch_size=64, sampler=test_sampler, drop_last=False,
    collate_fn=lambda batch: (dgl.batch([item[0] for item in batch]), torch.stack([item[1] for item in batch]))
)

quick_test_ratio = 0.1
num_quick_test = int(len(test_indices) * quick_test_ratio)
quick_test_indices = random.sample(test_indices, num_quick_test)

quick_test_sampler = SubsetRandomSampler(quick_test_indices)

quick_test_dataloader = GraphDataLoader(
    test_dataset, batch_size=2048, sampler=quick_test_sampler, drop_last=False,
    collate_fn=lambda batch: (dgl.batch([item[0] for item in batch]), torch.stack([item[1] for item in batch]))
)




#sageconv
num_classes = 5
hid_dim = 600
out_dim = 256
NUM_EPOCHS = 2000
l1_weight = 1e-4
# patience = 600         #max_enpoch
best_val_acc = 0.0
epochs_without_improvement = 0         # record
# train_losses = []
# ver_losses = []
# train_accuracies = []
# ver_accuracies = []
# test_accuracies = []

def l1_regularizer(model, l1_weight):
    l1_loss = torch.tensor(0.).to(device)
    for param in model.parameters():
        l1_loss += torch.norm(param, 2)
    return l1_weight * l1_loss

model = MyModel(hid_dim, out_dim, num_classes)
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=5e-4)
# optimizer = SGD(model.parameters(), lr=0.1, momentum=0.99, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)
# T_max = 200
# eta_min = 1e-6
# scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

x=0



for epoch in range(NUM_EPOCHS):
    model.train()
    x=x+1
    print(f'---------------------------------enpoch{x:.4f}-----------------------------------------')
    with open('./output/log1', 'a') as log_file:
        log_file.write(f'---------------------------------enpoch{x:.4f}-----------------------------------------' + '\n')
    for g, edge_labels in train_dataloader:
        g = g.to(device)
        edge_labels = edge_labels.to(device)

        optimizer.zero_grad()
        edge_scores= model(g, g.ndata['feature'])
        edge_scores = edge_scores.reshape(-1, num_classes)
        # loss = criterion(edge_scores, edge_labels.view(-1))
        ce_loss = criterion(edge_scores, edge_labels.view(-1))
        l1_reg_loss = l1_regularizer(model, l1_weight)

        loss = ce_loss + l1_reg_loss
        print(f'trainloss:  {loss:.4f}')
        with open('./output/log1', 'a') as log_file:
            log_file.write(f'trainloss:  {loss:.4f}' + '\n')
        # train_losses.append(loss)
        correct = 0
        total = 0
        _, predicted = torch.max(edge_scores.data, 1)
        total += edge_labels.size(0) * 15
        correct += (predicted == edge_labels.view(-1)).sum().item()
        accuracy = correct / total
        print(f'train Accuracy: {accuracy:.4f}')
        with open('./output/log1', 'a') as log_file:
            log_file.write(f'train Accuracy: {accuracy:.4f}' + '\n')
        # train_accuracies.append(accuracy)
        # print(loss)
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for g, edge_labels in quick_test_dataloader:
                g = g.to(device)
                edge_labels = edge_labels.to(device)

                edge_scores = model(g, g.ndata['feature'])
                edge_scores = edge_scores.reshape(-1, num_classes)
                loss = criterion(edge_scores, edge_labels.view(-1))
                print(f'verloss:   {loss:.4f}')
                with open('./output/log1', 'a') as log_file:
                    log_file.write(f'verloss:   {loss:.4f}' + '\n')
                # ver_losses.append(loss)
                # print(loss)

                _, predicted = torch.max(edge_scores.data, 1)
                total += edge_labels.size(0) * 15
                correct += (predicted == edge_labels.view(-1)).sum().item()
                accuracy = correct / total
                print(f'ver Accuracy: {accuracy:.4f}')
                with open('./output/log1', 'a') as log_file:
                    log_file.write(f'ver Accuracy: {accuracy:.4f}' + '\n')
                # ver_accuracies.append(accuracy)


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
            with open('./output/log1', 'a') as log_file:
                log_file.write(str(predicted)+'\n')

    print(correct)
    print(total)
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')
    with open('./output/log1', 'a') as log_file:
        log_file.write('correct:  ' + str(correct)+'\n')
        log_file.write('total:    ' + str(total) + '\n')
        log_file.write(f'Test Accuracy: {accuracy:.4f}'+'\n')
    # test_accuracies.append(accuracy)

    if accuracy > best_val_acc:
        best_val_acc = accuracy
        epochs_without_improvement = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        epochs_without_improvement += 1
        print(f'Epochs without improvement: {epochs_without_improvement}')
        with open('./output/log1', 'a') as log_file:
            log_file.write(f'Epochs without improvement: {epochs_without_improvement}' + '\n')
    # if epochs_without_improvement >= patience:
    #     print(f'Early stopping triggered at epoch {epoch}.')
    #     print(f'Best validation accuracy: {best_val_acc}')
    #     with open('./output/log', 'a') as log_file:
    #         log_file.write(f'Early stopping triggered at epoch {epoch}.' + '\n')
    #         log_file.write(f'Best validation accuracy: {best_val_acc}' + '\n')
    #     break
model.load_state_dict(torch.load('best_model.pth'))
print("Training completed. Loaded the best model.")
with open('./output/log1', 'a') as log_file:
    log_file.write("Training completed. Loaded the best model." + '\n')


# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# plt.plot(train_losses, label='Train Loss')
# plt.plot(ver_losses, label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('batch')
# plt.ylabel('Loss')
# plt.legend()
# plt.savefig('./output/res_loss.png')
#
# plt.subplot(1, 2, 2)
# plt.plot(train_accuracies, label='Train Accuracy')
# plt.plot(ver_accuracies, label='Validation Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.xlabel('batch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.savefig('./output/res_train_ver_acc.png')
#
# plt.figure()
# plt.plot(test_accuracies, label='Test Accuracy')
# plt.title('Test Accuracy Over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.savefig('./output/res_test_acc.png')
#
# plt.tight_layout()
# plt.savefig('./output/result.png')
# plt.show()















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