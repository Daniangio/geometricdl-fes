from collections import OrderedDict
from typing import Dict, Tuple, Union
from e3nn import o3
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import torch.optim
from torch_geometric.data.data import Data
from e3nn.math._soft_one_hot_linspace import soft_one_hot_linspace
from torch_scatter import scatter
import torch.nn.functional as F
from e3nn.nn import FullyConnectedNet
from models.messagepassing import MessagePassing, scatterd0
from e3nn.o3 import FullyConnectedTensorProduct


class Embedding(nn.Module):
    def __init__(
        self,
        irreps_node_input:  o3.Irreps,
        irreps_node_attr: o3.Irreps,
        irreps_edge_attr: o3.Irreps,
        irreps_node_out: o3.Irreps,
        lmax: int,
        mul: int,
        mpn_layers: int,
        max_radius: float,
        number_of_basis: int,
        num_neighbors: int
    ):
        super().__init__()
        self.lmax = lmax
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.num_neighbors = num_neighbors

        irreps_node_hidden = o3.Irreps([
            (mul // 4, (l, p))
            for l in range(lmax + 1)
            for p in [-1, 1]
        ])

        self.mp = MessagePassing(
            irreps_node_sequence=[irreps_node_input] + mpn_layers * [irreps_node_hidden] + [irreps_node_out],
            irreps_node_attr=irreps_node_attr,
            irreps_edge_attr=irreps_edge_attr,
            fc_neurons=[number_of_basis, mul, mul],
            num_neighbors=num_neighbors
        )
        
        self.irreps_node_out = self.mp.irreps_node_output
    
    def forward(self, data_edges: Union[Data, Dict[str, torch.Tensor]], data_bonds: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        for d in [data_edges, data_bonds]:
            if 'batch' in d:
                batch = d['batch']
            else:
                batch = d['pos'].new_zeros(d['pos'].shape[0], dtype=torch.long)

        node_attr = data_edges['node_attr']

        # The graph
        edge_src = data_edges['edge_index'][0]
        edge_dst = data_edges['edge_index'][1]

        # Edge attributes
        edge_vec = data_edges['pos'][edge_src] - data_edges['pos'][edge_dst]
        edge_sh = o3.spherical_harmonics(range(self.lmax + 1), edge_vec, True, normalization='component')
        edge_attr = torch.cat([data_edges['edge_attr'], edge_sh], dim=1)

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

        x = data_edges['x']
        x, edge_attr = self.mp(x, data_edges['node_attr'], edge_src, edge_dst, edge_attr, edge_length_embedding)
        return x, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding, batch


class EnergyPredictor(nn.Module):
    def __init__(
        self,
        irreps_node_input: o3.Irreps,
        irreps_node_attr: o3.Irreps,
        irreps_edge_attr: o3.Irreps,
        irreps_node_out: o3.Irreps,
        lmax: int,
        mul: int,
        mpn_layers: int,
        number_of_basis: int,
        num_neighbors: int
    ):
        super().__init__()

        self.num_neighbors = num_neighbors
        irreps_node_hidden_edges = o3.Irreps([
            (mul // 4, (l, p))
            for l in range(lmax + 1)
            for p in [-1, 1]
        ])

        self.mp = MessagePassing(
            irreps_node_sequence=[irreps_node_input] + mpn_layers * [irreps_node_hidden_edges] + [irreps_node_out],
            irreps_node_attr=irreps_node_attr,
            irreps_edge_attr=irreps_edge_attr,
            fc_neurons=[number_of_basis, mul, mul],
            num_neighbors=num_neighbors
        )
        irreps_mp_output = self.mp.irreps_node_output
        self.fctp = FullyConnectedTensorProduct(irreps_mp_output, irreps_edge_attr, irreps_node_out)
    
    def forward(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding, batch) -> torch.Tensor:
        node_features, edge_attr = self.mp(node_input, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding)
        x = self.fctp(node_features[edge_src], edge_attr)
        x = scatterd0(x, edge_dst, dim_size=node_features.shape[0]).div(self.num_neighbors**0.5)
        return scatter(x, batch, dim=0).div(self.num_neighbors**0.5)


class GeometricNetV2(LightningModule):
    def __init__(
        self,
        sample: Tuple[Data],
        lr: float,
        b1: float = 0.5,
        b2: float = 0.999,
        energy_levels: int = 40,
        lmax: int = 2,
        number_of_basis: int = 10,
        mpn_layers: int = 5,
        max_radius: float = 6.0,
        mul: int = 48,
        **kwargs
    ):
        super(GeometricNetV2, self).__init__()
        self.save_hyperparameters('lr', 'b1', 'b2')

        self.name = 'geometricv2'
        self.test_criterion_initials = 'acc'
        data_edges, data_bonds = sample
        self.num_nodes = data_edges.num_nodes / len(data_edges.e_label)

        irreps_node_input = o3.Irreps(f'{mul}x0e')
        irreps_node_attr = o3.Irreps(f'{data_edges.node_attr.size(1)}x0e')
        irreps_edge_attr = o3.Irreps('0e') + o3.Irreps.spherical_harmonics(lmax)
        irreps_embed_out = o3.Irreps(f'{mul}x0e+{mul}x0o+{mul//2}x1e+{mul//2}x1o+{mul//4}x2e+{mul//4}x2o')
        irreps_node_out = o3.Irreps(f'{mul*2}x0e')

        self.fc = FullyConnectedNet([data_edges.x.size(1), mul, mul], torch.sin)
        
        self.embedding = Embedding(
            irreps_node_input=irreps_node_input,
            irreps_node_attr=irreps_node_attr,
            irreps_edge_attr=irreps_edge_attr,
            irreps_node_out=irreps_embed_out,
            lmax=lmax,
            mul=mul,
            mpn_layers=1,
            max_radius=max_radius,
            number_of_basis=number_of_basis,
            num_neighbors=self.num_nodes
        )

        self.energy_predictor = EnergyPredictor(
            irreps_node_input=self.embedding.irreps_node_out,
            irreps_node_attr=irreps_node_attr,
            irreps_edge_attr=irreps_edge_attr,
            irreps_node_out=irreps_node_out,
            lmax=lmax,
            mul=mul,
            mpn_layers=mpn_layers,
            number_of_basis=number_of_basis,
            num_neighbors=self.num_nodes
        )

        self.lin1 = torch.nn.Linear(2*mul, 4*mul)
        self.lin2 = torch.nn.Linear(4*mul, 4*mul)
        self.out = torch.nn.Linear(4*mul, energy_levels)

    def forward(self, data_edges: Union[Data, Dict[str, torch.Tensor]], data_bonds: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        data_edges['x'] = self.fc(data_edges['x'])
        x, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding, batch = self.embedding(data_edges, data_bonds)
        x = self.energy_predictor(x, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding, batch)
        
        x = F.gelu(self.lin1(x))
        x = F.gelu(self.lin2(x))
        x = self.out(x)
        return F.softmax(x, dim=1), None
    
    def free_energy_loss(self, y_hat, y):
        delta = 1e-6
        logratio = torch.log((y_hat + delta) / (y + delta))
        batch_loss = torch.sum(y_hat * logratio, dim=1)
        return torch.sum(batch_loss) / len(y_hat) # F.mse_loss(y_hat, y)
    
    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_e = torch.optim.Adam(self.parameters(), lr=lr, betas=(b1, b2))
        # opt_g = torch.optim.Adam(list(self.embedding.parameters()) + list(self.generator.parameters()), lr=lr, betas=(b1, b2))
        return [opt_e], []
    
    def training_step(self, data, data_idx): # optimizer_idx <- add param if train with multiple optimizers
        data_edges, data_bonds = data
        # train energy predictor
        # if optimizer_idx == 0: <- use if multiple optimizers
        y_hat, _ = self(data_edges, data_bonds)
        e_loss = self.free_energy_loss(y_hat.squeeze(-1), data_edges.e_label)
        tqdm_dict = {"e_loss": e_loss.item()}
        try:
            self.log_dict(tqdm_dict)
        except:
            pass
        output = OrderedDict({"loss": e_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
        return output
    
    def test_step(self, data, data_idx):
        data_edges, data_bonds = data
        energy_predictions, energy_targets = [], []
        graph_indexes = []

        free_energy, _ = self(data_edges, data_bonds)
        for i in range(free_energy.size(0)):
            energy_predictions.append(torch.argmax(free_energy.squeeze(-1)[i]))
            energy_targets.append(torch.argmax(data_edges.e_label[i]))

            graph_indexes.extend(data_edges.graph_index[i])
        
        test_loss = self.free_energy_loss(free_energy.squeeze(-1), data_edges.e_label)
        test_acc = torch.sum((torch.tensor(energy_predictions) == torch.tensor(energy_targets))) / (len(energy_targets) * 1.0)
        data_edges.pos = None
        return {
            'test_acc': test_acc,
            'test_loss': test_loss,
            'energy_predictions': energy_predictions,
            'energy_targets': energy_targets,
            'graph_indexes': graph_indexes,
            'new_molecule': data_edges
        }
