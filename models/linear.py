import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import torch.optim

class LinearNet(LightningModule):
    def __init__(self, sample, lr, momentum, output_dim: int=1, hidden_dims: list=[256, 1024, 128, 64]):
        super(LinearNet, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nodes = len(sample.node_attr)
        self.nodes_features = sample.num_node_features
        self.lr = lr
        self.momentum = momentum
        self.criterion = nn.SmoothL1Loss(beta=1.0)
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.linear_layers = nn.ModuleList()

        # First hidden layer
        self.linear_layers.append(nn.Linear(self.nodes * self.nodes_features, hidden_dims[0]))
        last_dim = hidden_dims[0]

        # Remaining hidden layers
        for dim in hidden_dims[1:]:
            self.linear_layers.append(nn.Linear(last_dim, dim))
            self.linear_layers.append(nn.GELU())
            last_dim = dim
        
        # Output layer
        self.output = nn.Linear(last_dim, output_dim)

    def forward(self, samples):
        if not isinstance(samples, list):
            samples = [samples]

        bx = torch.stack([sample.x for sample in samples], 0)
        x = bx.view(bx.size(0), -1)
        x = self.linear_layers(x)

        return self.output(x)
    
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