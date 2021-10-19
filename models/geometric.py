from typing import Dict, Union
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct
from e3nn.nn._normact import NormActivation
from e3nn.nn.models.v2106.gate_points_message_passing import MessagePassing
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import torch.optim
from torch_geometric.data.data import Data
from e3nn.math._soft_one_hot_linspace import soft_one_hot_linspace
from torch_scatter import scatter
import torch.nn.functional as F
from e3nn.nn import FullyConnectedNet


class GeometricNet(LightningModule):
    def __init__(
        self,
        sample: Data,
        lr: float,
        b1: float,
        b2:float,
        output_dim: int = 20,
        lmax: int = 2,
        layers: int = 10,
        max_radius: float = 6.0,
        mul: int = 32,
        **kwargs
    ):
        super(GeometricNet, self).__init__()
        self.save_hyperparameters()
        self.name = 'geometric'
        self.num_nodes = sample.num_nodes
        self.lr = lr
        self.criterion = nn.BCELoss()
        self.test_criterion_initials = 'acc'

        self.embed = FullyConnectedNet(
            [sample.x.size(1), mul, mul],
            F.gelu
        )
        irreps_node_input = f"{mul}x0e"
        self.norm1 = NormActivation(irreps_node_input, torch.sigmoid)

        irreps_node_attr = '0e'
        irreps_edge_attr = o3.Irreps('0e') + o3.Irreps.spherical_harmonics(lmax)

        self.lmax = lmax
        self.max_radius = max_radius
        self.number_of_basis = 10

        irreps_node_hidden = o3.Irreps([
            (mul // 4, (l, p))
            for l in range(lmax + 1)
            for p in [-1, 1]
        ])

        self.mp1 = MessagePassing(
            irreps_node_sequence=[irreps_node_input] + layers * [irreps_node_hidden] + [irreps_node_hidden],
            irreps_node_attr=irreps_node_attr,
            irreps_edge_attr=irreps_edge_attr,
            fc_neurons=[self.number_of_basis, 2*mul],
            num_neighbors=self.num_nodes,
        )
        irreps_mp_output = self.mp1.irreps_node_output


        irreps_mid = o3.Irreps("64x0e + 24x1e + 24x1o + 16x2e + 16x2o")
        irreps_out = o3.Irreps(f"{4*mul}x0e")

        self.tp1 = FullyConnectedTensorProduct(
            irreps_in1=irreps_mp_output,
            irreps_in2=irreps_edge_attr,
            irreps_out=irreps_mid,
        )
        self.tp2 = FullyConnectedTensorProduct(
            irreps_in1=irreps_mid,
            irreps_in2=irreps_edge_attr,
            irreps_out=irreps_out,
        )
        self.norm2 = NormActivation(irreps_out, torch.sigmoid)

        self.lin1 = torch.nn.Linear(4*mul, 4*mul)
        self.lin2 = torch.nn.Linear(4*mul, 8*mul)
        self.out = torch.nn.Linear(8*mul, output_dim)

    def forward(self, data: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if 'batch' in data:
            batch = data['batch']
        else:
            batch = data['pos'].new_zeros(data['pos'].shape[0], dtype=torch.long)

        # The graph
        edge_src = data['edge_index'][0]
        edge_dst = data['edge_index'][1]

        # Edge attributes
        edge_vec = data['pos'][edge_src] - data['pos'][edge_dst]
        edge_sh = o3.spherical_harmonics(range(self.lmax + 1), edge_vec, True, normalization='component')
        edge_attr = torch.cat([data['edge_attr'], edge_sh], dim=1)

        # Edge length embedding
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = soft_one_hot_linspace(
            edge_length,
            0.0,
            self.max_radius,
            self.number_of_basis,
            basis='cosine',  # the cosine basis with cutoff = True goes to zero at max_radius
            cutoff=True,  # no need for an additional smooth cutoff
        ).mul(self.number_of_basis**0.5)

        # x (NODES, 14) node_attr (NODES) edge_src (EDGES) edge_dst (EDGES) edge_attr (EDGES, 1+3+5) edge_length_embedding (EDGES, number_of_basis)
        embed = self.embed(data['x'])
        embed = self.norm1(embed)
        x = self.mp1(embed, data['node_attr'], edge_src, edge_dst, edge_attr, edge_length_embedding)

        # For each edge, tensor product the features on the source node with the spherical harmonics
        edge_features = self.tp1(x[edge_src], edge_attr)
        x = scatter(edge_features, edge_dst, dim=0).div(self.num_nodes**0.5)

        edge_features = self.tp2(x[edge_src], edge_attr)
        edge_features = self.norm2(edge_features)
        x = scatter(edge_features, edge_dst, dim=0).div(self.num_nodes**0.5)
        x = scatter(x, batch, dim=0)

        x = F.gelu(self.lin1(x))
        x = F.gelu(self.lin2(x))
        return F.softmax(self.out(x), dim=1)
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
    
    def training_step(self, data, data_idx):
        y_hat = self(data)
        loss = self.criterion(y_hat.squeeze(-1), data.label)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, data, data_idx):
        batch_predictions, batch_targets = [], []

        y_hat = self(data)
        loss = self.criterion(y_hat.squeeze(-1), data.e_label)
        for i in range(y_hat.size(0)):
            batch_predictions.append(torch.argmax(y_hat.squeeze(-1)[i]))
            batch_targets.append(torch.argmax(data.e_label[i]))
        val_acc = torch.sum(torch.tensor(batch_predictions) == torch.tensor(batch_targets)) / (len(batch_targets) * 1.0)
        self.log('val_loss', loss)
        self.log('val_accuracy', val_acc)
        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss)
        return avg_loss
    
    def test_step(self, data, data_idx):
        energy_predictions, energy_targets = [], []
        graph_indexes = []

        y_hat = self(data)
        test_loss = self.criterion(y_hat.squeeze(-1), data.e_label)
        for i in range(y_hat.size(0)):
            energy_predictions.append(torch.argmax(y_hat.squeeze(-1)[i]))
            energy_targets.append(torch.argmax(data.e_label[i]))
            graph_indexes.append(data.graph_index[i])
        
        test_acc = torch.sum(torch.tensor(energy_predictions) == torch.tensor(energy_targets)) / (len(energy_targets) * 1.0)
        return {
            'test_acc': test_acc,
            'test_loss': test_loss,
            'energy_predictions': energy_predictions,
            'energy_targets': energy_targets,
            'graph_indexes': graph_indexes
        }
