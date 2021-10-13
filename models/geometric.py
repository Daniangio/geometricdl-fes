from typing import Dict, Union
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct
from e3nn.nn.models.v2106.gate_points_message_passing import MessagePassing
import torch
import torch.nn as nn
from e3nn.nn.models.v2103.gate_points_networks import NetworkForAGraphWithAttributes
from pytorch_lightning import LightningModule
import torch.optim
from torch_geometric.data.data import Data
from e3nn.math._soft_one_hot_linspace import soft_one_hot_linspace
from torch_scatter import scatter
import torch.nn.functional as F
from e3nn.nn import FullyConnectedNet

from test import ConvNetwork

class GeometricNet(LightningModule):
    def __init__(self, sample, lr, momentum, output_dim: int=1, lmax=2, layers=5, max_radius=6.0, mul=48):
        super(GeometricNet, self).__init__()
        self.name = 'geometric'
        self.num_nodes = sample.num_nodes
        self.lr = lr
        self.momentum = momentum
        self.criterion = nn.SmoothL1Loss(beta=1.0)

        self.embed = FullyConnectedNet(
            [8, mul, mul],
            F.gelu
        )

        irreps_node_input = f"{mul}x0o"
        irreps_node_attr = '0e'
        irreps_edge_attr = o3.Irreps('0e') + o3.Irreps.spherical_harmonics(lmax)

        self.lmax = lmax
        self.max_radius = max_radius
        self.number_of_basis = 10

        irreps_node_hidden = o3.Irreps([
            (mul, (l, p))
            for l in range(lmax + 1)
            for p in [-1, 1]
        ])

        self.mp1 = MessagePassing(
            irreps_node_sequence=[irreps_node_input] + layers * [irreps_node_hidden] + [irreps_node_hidden],
            irreps_node_attr=irreps_node_attr,
            irreps_edge_attr=irreps_edge_attr,
            fc_neurons=[self.number_of_basis, 2*mul],
            num_neighbors=2.0,
        )
        irreps_mp_output = self.mp1.irreps_node_output


        irreps_mid = o3.Irreps("64x0e + 24x1e + 24x1o + 16x2e + 16x2o")
        irreps_out = o3.Irreps(f"{mul//4}x0o+{mul//4}x0e+{mul//8}x1o+{mul//8}x1e+{mul//8}x2o+{mul//8}x2e")

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

        self.lin1 = torch.nn.Linear(2*(mul//4) + 2*3*(mul//8) + 2*5*(mul//8), 4*mul)
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

        # x (NODES, 5) node_attr (NODES) edge_src (EDGES) edge_dst (EDGES) edge_attr (EDGES, 1+3+5) edge_length_embedding (EDGES, number_of_basis)
        embed = self.embed(data['x'])
        x = self.mp1(embed, data['node_attr'], edge_src, edge_dst, edge_attr, edge_length_embedding)

        # For each edge, tensor product the features on the source node with the spherical harmonics
        edge_features = self.tp1(x[edge_src], edge_attr)
        x = scatter(edge_features, edge_dst, dim=0).div(self.num_nodes**0.5)

        edge_features = self.tp2(x[edge_src], edge_attr)
        x = scatter(edge_features, edge_dst, dim=0).div(self.num_nodes**0.5)
        x = scatter(x, batch, dim=0)

        x = F.gelu(self.lin1(x))
        x = F.gelu(self.lin2(x))
        return self.out(x)
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
    
    def training_step(self, data, data_idx):
        y_hat = self(data)
        # print(y_hat.squeeze(-1), data.label, y_hat.squeeze(-1).size(), data.label.size())
        loss = self.criterion(y_hat.squeeze(-1), data.label)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, data, data_idx):
        y_hat = self(data)
        loss = self.criterion(y_hat.squeeze(-1), data.label)
        self.log('val_loss', loss)
        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss)
        return avg_loss
    
    def test_step(self, data, data_idx):
        batch_predictions, batch_targets = [], []

        y_hat = self(data)
        loss = torch.mean(torch.abs(y_hat.squeeze(-1) - data.label))
        for pr, trg in zip(range(y_hat.size(0)), range(data.label.size(0))):
            batch_predictions.append((y_hat[pr].item(), data.graph_index[pr]))
            batch_targets.append((data.label[trg].item(), data.graph_index[pr]))
        return loss, batch_predictions, batch_targets


class OLDGeometricNet(LightningModule):
    def __init__(self, sample, lr, momentum, output_dim: int=1, irreps_node_output="64x0o+64x0e"):
        super(OLDGeometricNet, self).__init__()
        self.name = 'geometric'
        self.nodes = sample.x.size(0) // sample.label.size(0)
        self.lr = lr
        self.momentum = momentum
        self.criterion = nn.SmoothL1Loss(beta=1.0)
        self.output_dim = output_dim

        self.net = NetworkForAGraphWithAttributes(
        irreps_node_input="5x0e",
        irreps_node_attr="0e",
        irreps_edge_attr="0e",  # attributes in extra of the spherical harmonics
        irreps_node_output=irreps_node_output,
        max_radius=10.0,
        num_neighbors=3.0,
        num_nodes=9.0,
        pool_nodes = True
        )

        self.linear1 = torch.nn.Linear(self.net.irreps_node_output.dim, 256)
        self.nonlin1 = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.linear2 = torch.nn.Linear(256, 512)
        self.nonlin2 = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.linear3 = torch.nn.Linear(512, 1)

    def forward(self, data):
        node_outputs = self.net(data)
        x = self.nonlin1(self.linear1(node_outputs))
        x = self.nonlin2(self.linear2(x))
        return self.linear3(x)
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
    
    def training_step(self, data, data_idx):
        y_hat = self(data)
        print(y_hat.squeeze(-1), data.label, y_hat.squeeze(-1).size(), data.label.size())
        loss = self.criterion(y_hat.squeeze(-1), data.label)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, data, data_idx):
        y_hat = self(data)
        loss = self.criterion(y_hat.squeeze(-1), data.label)
        self.log('val_loss', loss)
        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss)
        return avg_loss
    
    def test_step(self, data, data_idx):
        batch_predictions, batch_targets = [], []

        y_hat = self(data)
        loss = torch.mean(torch.abs(y_hat.squeeze(-1) - data.label))
        for pr, trg in zip(range(y_hat.size(0)), range(data.label.size(0))):
            batch_predictions.append((y_hat[pr].item(), data.graph_index[pr]))
            batch_targets.append((data.label[trg].item(), data.graph_index[pr]))
        return loss, batch_predictions, batch_targets


from e3nn.nn.models.v2106.gate_points_message_passing import Convolution
from torch_cluster import radius_graph

def scatter(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    # special case of torch_scatter.scatter with dim=0
    out = src.new_zeros(dim_size, src.shape[1])
    index = index.reshape(-1, 1).expand_as(src)
    return out.scatter_add_(0, index, src)

class ConvNetwork(LightningModule):
    def __init__(
        self,
        sample, lr, momentum,
        irreps_in='8x0e',
        irreps_mid='16x0e + 16x0o + 16x1e + 16x1o',
        irreps_out='10x0e',
        min_radius=0.5,
        max_radius=1.5,
        num_neighbors=1.0,
        num_nodes=1.0,
        lmax=2
    ) -> None:
        super().__init__()
        self.name = 'geometric'
        self.lr = lr
        self.momentum = momentum
        self.criterion = nn.BCELoss()
        self.test_criterion_initials = 'bce'

        self.number_of_basis = 10
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.num_nodes = num_nodes
        self.lmax = lmax

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            Convolution(
            irreps_node_input=irreps_in,
            irreps_node_attr='0e',
            irreps_edge_attr= o3.Irreps.spherical_harmonics(lmax),
            irreps_node_output=irreps_mid,
            fc_neurons=[self.number_of_basis, 100],
            num_neighbors=num_neighbors
        )
        )
        for _ in range(10):
            self.convs.append(
                Convolution(
                irreps_node_input=irreps_mid,
                irreps_node_attr='0e',
                irreps_edge_attr= o3.Irreps.spherical_harmonics(lmax),
                irreps_node_output=irreps_mid,
                fc_neurons=[self.number_of_basis, 100],
                num_neighbors=num_neighbors
            )
            )
        self.convs.append(
            Convolution(
            irreps_node_input=irreps_mid,
            irreps_node_attr='0e',
            irreps_edge_attr= o3.Irreps.spherical_harmonics(lmax),
            irreps_node_output=irreps_out,
            fc_neurons=[self.number_of_basis, 100],
            num_neighbors=num_neighbors
        )
        )
    
    def preprocess(self, data):
        if 'batch' in data:
            batch = data['batch']
        else:
            batch = data['pos'].new_zeros(data['pos'].shape[0], dtype=torch.long)
        
        # Create graph
        edge_index = radius_graph(data['pos'], self.max_radius, batch)
        edge_src = edge_index[0]
        edge_dst = edge_index[1]

        # Edge attributes
        edge_vec = data['pos'][edge_src] - data['pos'][edge_dst]
        edge_attr = o3.spherical_harmonics(range(self.lmax + 1), edge_vec, True, normalization='component')

        # Edge length embedding
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = soft_one_hot_linspace(
            edge_length,
            self.min_radius,
            self.max_radius,
            self.number_of_basis,
            basis='gaussian',  # the smooth_finite basis with cutoff = True goes to zero at max_radius
            cutoff=True,  # no need for an additional smooth cutoff
        ).mul(self.number_of_basis**0.5)

        # Node attributes are not used here
        node_attr = data['x'].new_ones(data['x'].size(0), 1)
        return batch, data['x'], node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding
    
    def forward(self, data):
        batch, x, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding = self.preprocess(data)
        for conv in self.convs:
            x  = conv(x, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding)
        
        return F.sigmoid(scatter(x, batch, int(batch.max()) + 1).div(self.num_nodes**0.5))
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
    
    def training_step(self, data, data_idx):
        y_hat = self(data)
        loss = self.criterion(y_hat.squeeze(-1), data.label)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, data, data_idx):
        y_hat = self(data)
        loss = self.criterion(y_hat.squeeze(-1), data.label)
        self.log('val_loss', loss)
        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss)
        return avg_loss
    
    def test_step(self, data, data_idx):
        batch_predictions, batch_targets = [], []

        y_hat = self(data)
        print(y_hat.squeeze(-1)[:4], data.label[:4], self.criterion(y_hat.squeeze(-1), data.label))
        loss = torch.sum(self.criterion(y_hat.squeeze(-1), data.label))
        for pr, trg in zip(range(y_hat.size(0)), range(data.label.size(0))):
            batch_predictions.append((torch.argmax(y_hat.squeeze(-1)[trg]), data.graph_index[pr]))
            batch_targets.append((torch.argmax(data.label[trg]), data.graph_index[pr]))
        return loss, batch_predictions, batch_targets