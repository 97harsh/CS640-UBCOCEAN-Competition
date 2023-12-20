from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch
import torch_geometric.nn as nn
from torch_geometric.nn import global_max_pool, global_mean_pool
from torch.nn import Linear
from torch_geometric.nn.pool import SAGPooling



class GCN(torch.nn.Module):
    """GCN"""
    def __init__(self, dim_h, dataset):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, dim_h)
        self.conv2 = GCNConv(dim_h, dim_h)
        # self.conv3 = GCNConv(dim_h, dim_h)
        self.lin = Linear(dim_h, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # Node embeddings 
        h = self.conv1(x, edge_index)
        h = F.dropout(h, p=0.6, training=self.training)
        h = h.relu()
        h = self.conv2(h, edge_index)
        h = F.dropout(h, p=0.6, training=self.training)
        h = h.relu()
        # h = F.dropout(h, p=0.5, training=self.training)
        # h = self.conv3(h, edge_index)

        # Graph-level readout
        hG = global_mean_pool(h, batch)

        # Classifier
        h = F.dropout(hG, p=0.8, training=self.training)
        h = self.lin(h)

        return hG, F.log_softmax(h, dim=1)


class GCNSag(torch.nn.Module):
    """GCN"""
    def __init__(self, dim_h, dataset):
        super(GCNSag, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, dim_h)
        self.conv2 = GCNConv(dim_h, dim_h)
        self.sag = SAGPooling(in_channels=2*dim_h, ratio=0.1) # output to 1 final node
        # self.conv3 = GCNConv(dim_h, dim_h)
        self.lin = Linear(2*dim_h, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # Node embeddings 
        h1 = self.conv1(x, edge_index)
        h1 = F.dropout(h1, p=0.6, training=self.training)
        h1 = h1.relu()
        h2 = self.conv2(h1, edge_index)
        h2 = F.dropout(h2, p=0.6, training=self.training)
        h2 = h2.relu()
        # h = F.dropout(h, p=0.5, training=self.training)
        # h = self.conv3(h, edge_index)
        
        # Graph-level readout
        # out_sag = self.sag(torch.cat((h2,h1), dim=-1),edge_index, batch=batch)
        # hG=out_sag[0] # output from pooling
        # new_batch = out_sag[3]
        # hG = global_mean_pool(hG, new_batch)

        hG = global_mean_pool(torch.cat((h2,h1), dim=-1), batch)

        # Classifier
        # h = F.dropout(hG, p=0.8, training=self.training)
        h = self.lin(hG)

        return hG, F.log_softmax(h, dim=1)



class GCNTFEncoder(torch.nn.Module):
    """GCN"""
    def __init__(self, dim_h, dataset):
        super(GCNTFEncoder, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, dim_h)
        self.conv2 = GCNConv(dim_h, dim_h)
        tfencoder = torch.nn.TransformerEncoderLayer(2*dim_h, nhead=4)
        self.transformer_encoder = torch.nn.TransformerEncoder(tfencoder, num_layers=1)
        # self.conv3 = GCNConv(dim_h, dim_h)
        self.lin = Linear(2*dim_h, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # Node embeddings 
        h1 = self.conv1(x, edge_index)
        h1 = F.dropout(h1, p=0.6, training=self.training)
        h1 = h1.relu()
        h2 = self.conv2(h1, edge_index)
        h2 = F.dropout(h2, p=0.6, training=self.training)
        h2 = h2.relu()
        # breakpoint()

        mask = batch.unsqueeze(0) == batch.unsqueeze(1) # mask for each element corresponding to correct batch
        attention_mask = mask & (~torch.eye(batch.size(0), dtype=torch.bool).to(batch.device))
        ## Non self attention and attending only to intrabatch

        h3 = self.transformer_encoder(torch.cat((h2,h1), dim=-1), mask=attention_mask)
        hG = global_mean_pool(h3 , batch)

        # Classifier
        # h = F.dropout(hG, p=0.8, training=self.training)
        h = self.lin(hG)

        return hG, F.log_softmax(h, dim=1)


