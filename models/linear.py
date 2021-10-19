import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import torch.optim
import torch.nn.functional as F

class LinearNet(LightningModule):
    def __init__(self, sample, lr, output_dim: int=20, hidden_dims: list=[256, 1024, 128, 64]):
        super(LinearNet, self).__init__()
        self.name = 'linear'
        self.nodes = sample.num_nodes // sample.e_label.size(0)
        self.nodes_features = sample.edge_attr.size(1)
        self.lr = lr
        self.criterion = nn.BCELoss()
        self.test_criterion_initials = 'acc'
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

    def forward(self, data):
        x = data.edge_attr
        x = x.view(len(data.e_label), -1)
        for l in self.linears:
            x = l(x)

        return F.softmax(self.output(x), dim=1)
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.8)
    
    def training_step(self, data, data_idx):
        y_hat = self(data)
        print(y_hat.squeeze(), data.e_label)
        loss = self.criterion(y_hat.squeeze(), data.e_label)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, data, data_idx):
        y_hat = self(data)
        loss = self.criterion(y_hat.squeeze(), data.e_label)
        self.log('val_loss', loss)
        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss)
        return avg_loss
    
    def test_step(self, data, data_idx):
        energy_predictions, energy_targets = [], []
        graph_indexes = []

        y_hat = self(data)
        loss = torch.mean(torch.abs(y_hat - data.e_label))
        for i in range(y_hat.size(0)):
            energy_predictions.append(torch.argmax(y_hat[i]))
            energy_targets.append(torch.argmax(data.e_label[i]))
            graph_indexes.append(data.graph_index[i].item())
        
        test_acc = torch.sum(torch.tensor(energy_predictions) == torch.tensor(energy_targets)) / (len(energy_targets) * 1.0)
        return {
            'test_acc': test_acc,
            'test_loss': loss,
            'energy_predictions': energy_predictions,
            'energy_targets': energy_targets,
            'graph_indexes': graph_indexes
        }