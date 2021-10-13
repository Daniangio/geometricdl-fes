import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.nn.modules.linear import Linear
import torch.optim
import torch.nn.functional as F
from torch_geometric.data.data import Data
from torch_geometric.nn.conv import GraphConv
from torch_geometric.nn import EdgePooling
from torch_geometric.nn import (NNConv, graclus, max_pool_x, global_mean_pool)
from torch_geometric.utils import normalized_cut


class GraphConvPoolNet(LightningModule):
    def __init__(self, sample, lr, momentum, multipliers=[8, 16], channels=16):
        super(GraphConvPoolNet, self).__init__()
        self.name = 'graphdihedrals'
        self.edge_features = sample.edge_attr.size(1)
        self.lr = lr
        self.momentum = momentum
        self.criterion = nn.SmoothL1Loss(beta=1.0)
        self.test_criterion_initials = 'mae'
        self.channels = channels

        nn1 = nn.Sequential(nn.Linear(self.edge_features, 64), nn.ReLU(),
                            nn.Linear(64, sample.num_features * 64))
        self.conv1 = NNConv(sample.num_features, 64, nn1, aggr='mean')

        nn2 = nn.Sequential(nn.Linear(self.edge_features, 64), nn.ReLU(),
                            nn.Linear(64, 64 * 64))
        self.conv2 = NNConv(64, 64, nn2, aggr='mean')

        self.fc1 = nn.Linear(64, 512)
        self.fc2 = nn.Linear(512, 512)
        self.final = nn.Linear(512, 1)

    def forward(self, data: Data):
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut(data.edge_index, torch.ones((data.edge_attr.size(0)), device=data.edge_attr.device), num_nodes=data.num_nodes)
        cluster = graclus(data.edge_index, weight, data.num_nodes)
        x, batch = max_pool_x(cluster, data.x, data.batch)

        x = global_mean_pool(x, batch)
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        return self.final(x).view(-1)
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
    
    def training_step(self, data, data_idx):
        y_hat = self(data)
        loss = self.criterion(y_hat, data.label)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, data, data_idx):
        y_hat = self(data)
        loss = self.criterion(y_hat, data.label)
        self.log('val_loss', loss)
        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss)
        return avg_loss
    
    def test_step(self, data, data_idx):
        batch_predictions, batch_targets = [], []

        y_hat = self(data)
        loss = torch.mean(torch.abs(y_hat - data.label))
        for pr, trg in zip(range(y_hat.size(0)), range(data.label.size(0))):
            batch_predictions.append((y_hat[pr].item(), data.graph_index[pr]))
            batch_targets.append((data.label[trg].item(), data.graph_index[pr]))
        return loss, batch_predictions, batch_targets


class BISGraphConvPoolNet(LightningModule):
    def __init__(self, sample, lr, momentum, multipliers=[4, 4, 4], channels=16):
        super(BISGraphConvPoolNet, self).__init__()
        self.name = 'graphdihedrals'
        self.nodes_features = sample.x.size(1)
        self.lr = lr
        self.momentum = momentum
        self.criterion = nn.SmoothL1Loss(beta=1.0)
        self.test_criterion_initials = 'mae'
        self.channels = channels
        self.input = GraphConv(sample.num_node_features, self.channels)
        self.conv1 = GraphConv(self.channels, multipliers[0] * self.channels)
        self.pool1 = EdgePooling(multipliers[0] * self.channels, dropout=0.2)
        self.conv2 = GraphConv(multipliers[0] * self.channels, (multipliers[1] + multipliers[0]) * self.channels)
        self.pool2 = EdgePooling((multipliers[1] + multipliers[0]) * self.channels, dropout=0.2)
        self.conv3 = GraphConv((multipliers[1] + multipliers[0]) * self.channels,
                               (multipliers[2] + multipliers[1] + multipliers[0]) * self.channels)

        self.looppool = EdgePooling((multipliers[2] + multipliers[1] + multipliers[0]) * self.channels, dropout=0.2)
        self.loopconv = GraphConv((multipliers[2] + multipliers[1] + multipliers[0]) * self.channels,
                                  (multipliers[2] + multipliers[1] + multipliers[0]) * self.channels)

        # Readout layer
        self.readout = max_pool_x
        self.finalnodes = 2
        self.output1 = nn.Linear(self.finalnodes * (multipliers[2] + multipliers[1] + multipliers[0]) * self.channels, 128)
        self.output2 = nn.Linear(128, 256)
        self.output3 = nn.Linear(256, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.input(x, edge_index)
        x = F.gelu(x)
        x = self.conv1(x, edge_index)
        x = F.gelu(x)
        
        batch = torch.tensor([0 for _ in x], dtype=torch.long, device=x.device)
        (x, edge_index, b, unpool_info) = self.pool1(x, edge_index, batch=batch)

        x = self.conv2(x, edge_index)
        x = F.gelu(x)
        batch = torch.tensor([0 for _ in x], dtype=torch.long, device=x.device)
        (x, edge_index, b, unpool_info) = self.pool2(x, edge_index, batch=batch)

        x = self.conv3(x, edge_index)
        x = F.gelu(x)

        # Readout and output
        cluster = torch.as_tensor([i % self.finalnodes for i in range(len(x))], device=x.device)
        (x, cluster) = self.readout(cluster, x, batch)

        x = F.gelu(self.output1(x.view(-1)))
        x = F.gelu(self.output2(x))
        return self.output3(x)
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
    
    def training_step(self, data, data_idx):
        y_hat = self(data)
        loss = self.criterion(y_hat, data.label)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, data, data_idx):
        y_hat = self(data)
        loss = self.criterion(y_hat, data.label)
        self.log('val_loss', loss)
        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss)
        return avg_loss
    
    def test_step(self, data, data_idx):
        batch_predictions, batch_targets = [], []

        y_hat = self(data)
        loss = torch.mean(torch.abs(y_hat - data.label))
        for pr, trg in zip(range(y_hat.size(0)), range(data.label.size(0))):
            batch_predictions.append((y_hat[pr].item(), data.graph_index[pr]))
            batch_targets.append((data.label[trg].item(), data.graph_index[pr]))
        return loss, batch_predictions, batch_targets