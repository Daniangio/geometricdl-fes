from collections import OrderedDict
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


class Embedding(nn.Module):
    def __init__(
        self,
        input_dim: int,
        irreps_node_attr: o3.Irreps,
        irreps_edge_attr: o3.Irreps,
        irreps_node_out: o3.Irreps,
        lmax: int,
        mul: int,
        mpn_layers: int,
        max_radius: float,
        number_of_basis: int,
        num_neighbors: int,
    ):
        super().__init__()
        self.lmax = lmax
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis

        self.embed = FullyConnectedNet(
            [input_dim, mul, mul],
            F.gelu
        )
        irreps_node_embed = o3.Irreps(f'{mul}x0e')

        irreps_node_hidden = o3.Irreps([
            (mul // 4, (l, p))
            for l in range(lmax + 1)
            for p in [-1, 1]
        ])

        self.mp1 = MessagePassing(
            irreps_node_sequence=[irreps_node_embed] + mpn_layers * [irreps_node_hidden] + [irreps_node_out],
            irreps_node_attr=irreps_node_attr,
            irreps_edge_attr=irreps_edge_attr,
            fc_neurons=[number_of_basis, mul, 2*mul, 2*mul],
            num_neighbors=num_neighbors,
        )
        self.irreps_node_out = self.mp1.irreps_node_output
    
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
        embed = self.embed(data['x'])
        embed = self.mp1(embed, data['node_attr'], edge_src, edge_dst, edge_attr, edge_length_embedding)
        return embed, data['node_attr'], edge_src, edge_dst, edge_attr, edge_length_embedding, batch


class EnergyPredictor(nn.Module):
    def __init__(
        self,
        irreps_node_input: o3.Irreps,
        irreps_node_attr: o3.Irreps,
        irreps_edge_attr: o3.Irreps,
        energy_levels: int,
        lmax: int,
        mul: int,
        mpn_layers: int,
        number_of_basis: int,
        num_neighbors: int
    ):
        super().__init__()

        irreps_node_hidden = o3.Irreps([
            (mul // 4, (l, p))
            for l in range(lmax + 1)
            for p in [-1, 1]
        ])
        irreps_node_out = o3.Irreps([(energy_levels, (0, 1))])

        self.mp1 = MessagePassing(
            irreps_node_sequence=[irreps_node_input] + mpn_layers * [irreps_node_hidden] + [irreps_node_out],
            irreps_node_attr=irreps_node_attr,
            irreps_edge_attr=irreps_edge_attr,
            fc_neurons=[number_of_basis, mul, 2*mul, 2*mul],
            num_neighbors=num_neighbors,
        )
        irreps_mp_output = self.mp1.irreps_node_output
    
    def forward(self, x, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding, batch) -> torch.Tensor:
        x = self.mp1(x, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding)
        x = scatter(x, batch, dim=0)
        return F.softmax(x, dim=1)


class DihedralsPredictor(nn.Module):
    def __init__(
        self,
        irreps_node_input: o3.Irreps,
        irreps_node_attr: o3.Irreps,
        irreps_edge_attr: o3.Irreps,
        num_dihedrals: int,
        lmax: int,
        mul: int,
        mpn_layers: int,
        number_of_basis: int,
        num_neighbors: int
    ):
        super().__init__()

        irreps_node_hidden = o3.Irreps([
            (mul // 4, (l, p))
            for l in range(lmax + 1)
            for p in [-1, 1]
        ])
        irreps_node_out = o3.Irreps([(mul, (0, 1))])

        self.mp1 = MessagePassing(
            irreps_node_sequence=[irreps_node_input] + mpn_layers * [irreps_node_hidden] + [irreps_node_out],
            irreps_node_attr=irreps_node_attr,
            irreps_edge_attr=irreps_edge_attr,
            fc_neurons=[number_of_basis, mul, 2*mul, 2*mul],
            num_neighbors=num_neighbors,
        )
        irreps_mp_output = self.mp1.irreps_node_output

        self.fc1 = nn.Linear(mul, mul*4)
        self.fc2 = nn.Linear(mul*4, mul*4)
        self.out = nn.Linear(mul*4, num_dihedrals)


    
    def forward(self, x, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding, batch) -> torch.Tensor:
        x = self.mp1(x, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding)
        x = scatter(x, batch, dim=0)

        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        return F.softmax(self.out(x), dim=1)


class GeometricPhiPsiNet(LightningModule):
    def __init__(
        self,
        sample: Data,
        lr: float,
        b1: float = 0.5,
        b2: float = 0.999,
        energy_levels: int = 20,
        num_dihedrals: int = 2,
        lmax: int = 2,
        number_of_basis: int = 10,
        mpn_layers: int = 4,
        max_radius: float = 6.0,
        mul: int = 32,
        **kwargs
    ):
        super(GeometricPhiPsiNet, self).__init__()
        self.save_hyperparameters('lr', 'b1', 'b2')

        self.name = 'geometric-phi-psi'
        self.test_criterion_initials = 'acc'
        self.num_nodes = sample.num_nodes

        irreps_node_attr = o3.Irreps(f'{sample.node_attr.size(1)}x0e')
        irreps_edge_attr = o3.Irreps('0e') + o3.Irreps.spherical_harmonics(lmax)
        irreps_embed_out = o3.Irreps(f'{mul}x0e+{mul}x0o+{mul//4}x1o+{mul//8}x2o')
        
        self.embedding = Embedding(
            input_dim=sample.x.size(1),
            irreps_node_attr=irreps_node_attr,
            irreps_edge_attr=irreps_edge_attr,
            irreps_node_out=irreps_embed_out,
            lmax=lmax,
            mul=mul,
            mpn_layers=mpn_layers,
            max_radius=max_radius,
            number_of_basis=number_of_basis,
            num_neighbors=sample.num_nodes
        )

        self.energy_predictor = EnergyPredictor(
            irreps_node_input=self.embedding.irreps_node_out,
            irreps_node_attr=irreps_node_attr,
            irreps_edge_attr=irreps_edge_attr,
            energy_levels=energy_levels,
            lmax=lmax,
            mul=mul,
            mpn_layers=mpn_layers,
            number_of_basis=number_of_basis,
            num_neighbors=sample.num_nodes
        )

        self.dihedrals_predictor = DihedralsPredictor(
            irreps_node_input=self.embedding.irreps_node_out,
            irreps_node_attr=irreps_node_attr,
            irreps_edge_attr=irreps_edge_attr,
            num_dihedrals=num_dihedrals,
            lmax=lmax,
            mul=mul,
            mpn_layers=mpn_layers // 2,
            number_of_basis=number_of_basis,
            num_neighbors=sample.num_nodes
        )

    def forward(self, data: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        x, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding, batch = self.embedding(data)
        return (
            self.energy_predictor(x, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding, batch),
            self.dihedrals_predictor(x, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding, batch)
        )
    
    def energy_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)
    
    def dihedral_loss(self, y_hat, y):
        return F.mse_loss(y_hat, y)
    
    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_e = torch.optim.Adam(self.energy_predictor.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.dihedrals_predictor.parameters(), lr=lr, betas=(b1, b2))
        return [opt_e, opt_d], []
    
    def training_step(self, data, data_idx, optimizer_idx):
        x, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding, batch = self.embedding(data)

        # train energy predictor
        if optimizer_idx == 0:
            y_hat = self.energy_predictor(x, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding, batch)
            e_loss = self.energy_loss(y_hat.squeeze(-1), data.e_label)
            tqdm_dict = {"e_loss": e_loss.item()}
            self.log_dict(tqdm_dict)
            output = OrderedDict({"loss": e_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output
        # train dihedrals predictor
        if optimizer_idx == 1:
            y_hat = self.dihedrals_predictor(x, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding, batch)
            d_loss = self.dihedral_loss(y_hat.squeeze(-1), data.d_label)
            tqdm_dict = {"d_loss": d_loss.item()}
            self.log_dict(tqdm_dict)
            output = OrderedDict({"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output
    
    def test_step(self, data, data_idx):
        dihedrals_predictions, dihedrals_targets, energy_predictions, energy_targets = [], [], [], []
        graph_indexes = []

        y_hat_e, y_hat_d = self(data)
        for i in range(y_hat_d.size(0)):
            dihedrals_predictions.append(y_hat_d.squeeze(-1)[i].tolist())
            dihedrals_targets.append(data.d_label[i].tolist())

            energy_predictions.append(torch.argmax(y_hat_e.squeeze(-1)[i]))
            energy_targets.append(torch.argmax(data.e_label[i]))

            graph_indexes.append(data.graph_index[i])
        
        test_loss = self.energy_loss(y_hat_e.squeeze(-1), data.e_label)
        test_acc = torch.sum((torch.tensor(energy_predictions) == torch.tensor(energy_targets))) / (len(energy_targets) * 1.0)
        return {
            'test_acc': test_acc,
            'test_loss': test_loss,
            'dihedrals_predictions': dihedrals_predictions,
            'dihedrals_targets': dihedrals_targets,
            'energy_predictions': energy_predictions,
            'energy_targets': energy_targets,
            'graph_indexes': graph_indexes
        }
