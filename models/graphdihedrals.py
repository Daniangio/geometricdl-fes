import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import torch.optim
import torch.nn.functional as F
from torch_geometric.nn.conv import GraphConv
from torch_geometric.nn import EdgePooling
from torch_scatter import scatter_mean


class GraphConvPoolNet(LightningModule):
    def __init__(self, sample, lr, momentum, multipliers=[4, 4, 4], channels=16):
        super(GraphConvPoolNet, self).__init__()
        self.name = 'graphdihedrals'
        self.nodes_features = sample.x.size(1)
        self.lr = lr
        self.momentum = momentum
        self.criterion = nn.SmoothL1Loss(beta=1.0)
        self.channels = channels
        self.input = GraphConv(self.nodes_features, self.channels)
        self.conv1 = GraphConv(self.channels, multipliers[0] * self.channels)
        self.pool1 = EdgePooling(multipliers[0] * self.channels, dropout=0.2)
        self.conv2 = GraphConv(multipliers[0] * self.channels, (multipliers[1] + multipliers[0]) * self.channels)
        self.pool2 = EdgePooling((multipliers[1] + multipliers[0]) * self.channels, dropout=0.2)
        self.conv3 = GraphConv((multipliers[1] + multipliers[0]) * self.channels,
                               (multipliers[2] + multipliers[1] + multipliers[0]) * self.channels)

        self.looppool = EdgePooling((multipliers[2] + multipliers[1] + multipliers[0]) * self.channels, dropout=0.2)
        self.loopconv = GraphConv((multipliers[2] + multipliers[1] + multipliers[0]) * self.channels,
                                  (multipliers[2] + multipliers[1] + multipliers[0]) * self.channels)

        self.fc1 = nn.Linear((multipliers[2] + multipliers[1] + multipliers[0]) * self.channels, 128)
        self.fc2 = nn.Linear(128, 64)
        self.final = nn.Linear(64, 1)

    def forward(self, sample):
        x, edge_index, batch = sample.x, sample.edge_index, sample.batch
        x = F.gelu(self.input(x, edge_index))
        x = F.gelu(self.conv1(x, edge_index))
        (x, edge_index, batch, unpool_info) = self.pool1(x, edge_index, batch=batch)

        x = F.gelu(self.conv2(x, edge_index))
        (x, edge_index, batch, unpool_info) = self.pool2(x, edge_index, batch=batch)

        x = F.gelu(self.conv3(x, edge_index))

        (x, edge_index, batch, unpool_info) = self.looppool(x, edge_index, batch=batch)
        x = F.gelu(self.loopconv(x, edge_index))

        # Readout and output
        x = scatter_mean(x, batch, dim=0)
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
        loss = self.criterion(y_hat, data.label)
        for pr, trg in zip(range(y_hat.size(0)), range(data.label.size(0))):
            batch_predictions.append((y_hat[pr].item(), data.graph_index[pr]))
            batch_targets.append((data.label[trg].item(), data.graph_index[pr]))
        return loss, batch_predictions, batch_targets





