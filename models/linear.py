import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import torch.optim

class LinearNet(LightningModule):
    def __init__(self, sample, lr, momentum, output_dim: int=1, hidden_dims: list=[256, 1024, 128, 64]):
        super(LinearNet, self).__init__()
        self.name = 'linear'
        self.nodes = sample.edge_attr.size(0) // sample.label.size(0)
        self.nodes_features = sample.edge_attr.size(1)
        self.lr = lr
        self.momentum = momentum
        self.criterion = nn.SmoothL1Loss(beta=1.0)
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        linear_layers = []

        # First hidden layer
        linear_layers.append(nn.Linear(self.nodes * self.nodes_features, hidden_dims[0]))
        last_dim = hidden_dims[0]

        # Remaining hidden layers
        for dim in hidden_dims[1:]:
            linear_layers.append(nn.Linear(last_dim, dim))
            linear_layers.append(nn.GELU())
            last_dim = dim
        
        self.linears = nn.ModuleList(linear_layers)
        
        # Output layer
        self.output = nn.Linear(last_dim, output_dim)

    def forward(self, sample):
        x = sample.edge_attr
        x = x.view(len(sample.label), -1)
        for l in self.linears:
            x = l(x)

        return self.output(x)
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
    
    def training_step(self, data, data_idx):
        y_hat = self(data)
        loss = self.criterion(y_hat.squeeze(), data.label)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, data, data_idx):
        y_hat = self(data)
        loss = self.criterion(y_hat.squeeze(), data.label)
        self.log('val_loss', loss)
        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss)
        return avg_loss
    
    def test_step(self, data, data_idx):
        batch_predictions, batch_targets = [], []

        y_hat = self(data)
        loss = self.criterion(y_hat.squeeze(), data.label)
        for pr, trg in zip(range(y_hat.size(0)), range(data.label.size(0))):
            batch_predictions.append((y_hat[pr].item(), data.graph_index[pr]))
            batch_targets.append((data.label[trg].item(), data.graph_index[pr]))
        return loss, batch_predictions, batch_targets