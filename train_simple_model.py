from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch
import torch_geometric.nn as nn
import torch_geometric
from torch_geometric.loader import DataLoader
import os
from torch_geometric.data import Data, Dataset
import random
from torch_geometric.utils import k_hop_subgraph
from models import GCN, GCNSag, GCNTFEncoder
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

CLASS_COUNTS = {'EC':99, 'HGSC': 177, 'MC':37, 'CC':79, 'LGSC':38}

def sample_k_hop_subgraph(G, start_node, k=5):
    """ 
    This functions returns a subgraph centeed around start_node and all neighbors upto k hops away
    """
    
    subset_nodes,_,_,_ = k_hop_subgraph(start_node,k,G.edge_index)

    return G.subgraph(subset_nodes)

class GraphDataset(Dataset):
    """
    input and label image dataset
    Reads adj matrix and creates the adj lists
    """

    def __init__(self, root, ids, num_levels_sampled=8, node_drop_prob=0.8, train=False):
        super(GraphDataset, self).__init__()
        """
        Args:

        fileDir(string):  directory with all the input images.
        transform(callable, optional): Optional transform to be applied on a sample
        """
        self.root = root
        self.ids = ids
        self.classdict = {'EC':0, 'HGSC':1, 'MC':2, 'CC':3, 'LGSC':4}
        self.num_levels_sampled = num_levels_sampled
        self.node_drop_prob = node_drop_prob
        self.train=train


    def get(self, index):
        sample = {}
        info = self.ids[index].replace('\n', '')
        file_name, label = info.split(',')[0].rsplit('.', 1)[0], info.split(',')[1]
        # site, file_name = file_name.split('/')

        sample['label'] = self.classdict[label]
        sample['id'] = file_name

        data_path = os.path.join(self.root, f'{file_name}.pt')

        if os.path.exists(data_path):
            data = torch.load(data_path, map_location=lambda storage, loc: storage)
        else:
            raise ValueError("File is incorrect")


        y_one_hot = torch.zeros(self.num_classes)
        y_one_hot[self.classdict[label]] = 1
        data.y = y_one_hot.unsqueeze(0)
        if 'edge_latent' in data:
            del data['edge_latent']
        if 'centroid' in data:
            del data['centroid']
        # Same as selecting a small patch of image and trying to predict
        if self.train: # if this is training
            new_data = sample_k_hop_subgraph(data, random.randint(0,data.num_nodes-1), self.num_levels_sampled)
            return new_data
        else:
            return data
        # return Data(x = features, edge_index=edge_indices, num_nodes=num_nodes, y=y_one_hot.unsqueeze(0))

    @property
    def num_classes(self):
        return len(self.classdict.keys())

    def len(self):
        return len(self.ids)



def train(model, train_loader, val_loader, test_loader, lr=5e-3):
    class_weights = torch.tensor([1/x for x in CLASS_COUNTS.values()]).to(device)
    class_weights = F.normalize(class_weights, dim=0)
    criterion = torch.nn.CrossEntropyLoss(class_weights, label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(),
                                      lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                            mode='min',
                                                            factor=0.5,
                                                            patience=5)                                
    epochs = 5
    highest_val_acc=0
    model.train()
    for epoch in range(epochs+1):
        total_loss = 0
        acc = 0
        val_loss = 0
        val_acc = 0
        train_labels = []
        train_outputs = []

        # Train on batches
        for data in train_loader:
          optimizer.zero_grad()
          data = data.to(device)
          _, out = model(data.x, data.edge_index, data.batch)
          loss = criterion(out, data.y)
          total_loss += loss / len(train_loader)
          train_labels.append(data.y.detach().cpu().numpy().argmax(1))
          train_outputs.append(out.detach().cpu().numpy().argmax(1))
        #   acc += accuracy(out.argmax(dim=1), data.y.argmax(dim=1)) / len(train_loader)
          loss.backward()
          optimizer.step()
        train_labels = np.concatenate(train_labels)
        train_outputs = np.concatenate(train_outputs)
        acc = accuracy(train_outputs, train_labels)

        # Validation
        val_loss, val_acc = test(model, val_loader)
        if highest_val_acc<val_acc:
            highest_val_acc = val_acc
        scheduler.step(val_loss)
        # Print metrics every epoch
        # if(epoch % 30 == 0):
        #     print(f'Epoch {epoch:>3} | Train Loss: {total_loss:.2f} '
        #         f'| Train Acc: {acc*100:>5.2f}% '
        #         f'| Val Loss: {val_loss:.2f} '
        #         f'| Val Acc: {val_acc*100:.2f}%')

    test_loss, test_acc = test(model, test_loader)
    print(f'Test Loss: {test_loss:.2f} | Test Acc: {test_acc*100:.2f}%')
    
    return model, test_acc, highest_val_acc

from sklearn.metrics import confusion_matrix
import numpy as np

@torch.no_grad()
def test(model, loader):

    class_weights = torch.tensor([1/x for x in CLASS_COUNTS.values()]).to(device)
    class_weights = F.normalize(class_weights, dim=0)
    criterion = torch.nn.CrossEntropyLoss(class_weights, label_smoothing=0.1)

    model.eval()
    loss = 0 
    acc = 0

    y_true = []
    y_pred = []

    for data in loader:
        data = data.to(device)
        _, out = model(data.x, data.edge_index, data.batch)
        loss += criterion(out, data.y) / len(loader)

        # acc += accuracy(out.argmax(dim=1), data.y.argmax(dim=1)) / len(loader)

        # Save true and predicted labels
        y_true.append(data.y.argmax(dim=1).cpu().numpy())
        y_pred.append(out.argmax(dim=1).cpu().numpy())
    # Calculate confusion matrix
    y_true = np.concatenate(y_true) 
    y_pred = np.concatenate(y_pred)
    acc = accuracy(y_pred, y_true)
    conf_mat = confusion_matrix(y_true, y_pred)

    # print("Validation Confusion Matrix:")
    # print(conf_mat)

    return loss, acc


def accuracy(pred_y, y):
    """Calculate accuracy."""
    def send_cpu(x):
        if torch.is_tensor(x):
            return x.cpu()
        else:
            return x

    return balanced_accuracy_score(send_cpu(y),send_cpu(pred_y))


fold = 2
data_root="/projectnb/cs640grp/students/hsharma/UBC_processed/ctranspath_embed"
train_set = f"/projectnb/cs640grp/students/hsharma/5folds/train_fold_{fold}.txt"
val_set = f"/projectnb/cs640grp/students/hsharma/5folds/val_fold_{fold}.txt"
test_set = f"/projectnb/cs640grp/students/hsharma/5folds/test_fold_{fold}.txt"

ids_train = open(train_set).readlines()
ids_val = open(val_set).readlines()
ids_test = open(test_set).readlines()

train_dataset = GraphDataset(data_root, ids_train)
val_dataset = GraphDataset(data_root, ids_val)
test_dataset = GraphDataset(data_root, ids_test)
print("Data Loaded")
# Create a PyTorch Geometric DataLoader for your dataset
batch_size = 3
num_workers = 4
dataloader_train = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=num_workers, shuffle=True,
                              pin_memory=True, drop_last=True)
dataloader_val = DataLoader(val_dataset, batch_size=batch_size,
                            num_workers=num_workers, shuffle=True,
                            pin_memory=True, drop_last=True)
dataloader_test = DataLoader(test_dataset, batch_size=batch_size,
                             num_workers=num_workers, shuffle=True,
                             pin_memory=True, drop_last=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# gcn = GCN(dim_h=32, dataset=train_dataset).to(device)

lr_rate = [ 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
hidden_size=[16,32,64,128,256]

df = pd.DataFrame(columns=["fold","lr_rate","dim_hidden","test_accuracy"])
highest_test_acc = 0
for lr in lr_rate:
    for dim_h in hidden_size:
        for _ in range(3): # running 3 times, because training is unstable
            gcn = GCNTFEncoder(dim_h=dim_h, dataset=train_dataset).to(device) 
            gcn, test_acc, highest_val_acc = train(gcn, dataloader_train, dataloader_val, dataloader_test, lr=lr)
            if round(test_acc,2)>highest_test_acc:
                torch.save(gcn, f"fold{fold}_best_model.pt")
                highest_test_acc = round(test_acc,2)
            if pd.isna(df.index.max()):
                df.loc[0] = [fold, lr, dim_h, test_acc]
            else:
                df.loc[df.index.max() +1] = [2, lr, dim_h, test_acc]
            df.to_csv(f"fold{fold}_scores.csv",index=False)
print(df)
# df.to_csv("fold3_scores.csv",index=False)

