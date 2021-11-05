from collections import OrderedDict
from typing import Dict, Union
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct
from e3nn.nn.models.v2106.gate_points_message_passing import MessagePassing
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import torch.optim
from torch_geometric.data.data import Data
from e3nn.math._soft_one_hot_linspace import soft_one_hot_linspace
from torch_scatter import scatter
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from e3nn.nn import FullyConnectedNet


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
        num_neighbors: int,
        use_forces: bool
    ):
        super().__init__()
        self.lmax = lmax
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis

        irreps_node_hidden = o3.Irreps([
            (mul // 4, (l, p))
            for l in range(lmax + 1)
            for p in [-1, 1]
        ])

        self.mp = MessagePassing(
            irreps_node_sequence=[irreps_node_input] + mpn_layers * [irreps_node_hidden] + [irreps_node_out],
            irreps_node_attr=irreps_node_attr,
            irreps_edge_attr=irreps_edge_attr,
            fc_neurons=[number_of_basis, mul, 2*mul],
            num_neighbors=num_neighbors,
        )
        self.irreps_node_out = self.mp.irreps_node_output
        self.use_forces = use_forces
    
    def forward(self, data_edges: Union[Data, Dict[str, torch.Tensor]], data_bonds: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        for d in [data_edges, data_bonds]:
            if 'batch' in d:
                batch = d['batch']
            else:
                batch = d['pos'].new_zeros(d['pos'].shape[0], dtype=torch.long)

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

        # # Bonds length embedding
        # bonds_vec = data_bonds['pos'][bonds_src] - data_bonds['pos'][bonds_dst]
        # bonds_length = bonds_vec.norm(dim=1)
        # bonds_length_embedding = soft_one_hot_linspace(
        #     bonds_length,
        #     0.0,
        #     self.max_radius,
        #     self.number_of_basis,
        #     basis='cosine',  # the cosine basis with cutoff = True goes to zero at max_radius
        #     cutoff=True,  # no need for an additional smooth cutoff
        # ).mul(self.number_of_basis**0.5)

        if self.use_forces:
            x = torch.cat([data_edges['x'], data_edges['forces']], dim=1)
        else:
            x = data_edges['x']
        embed_on_edges = self.mp(x, data_edges['node_attr'], edge_src, edge_dst, edge_attr, edge_length_embedding)
        return embed_on_edges, data_edges['node_attr'], edge_src, edge_dst, edge_attr, edge_length_embedding, batch


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

        irreps_node_hidden_edges = o3.Irreps([
            (mul // 4, (l, p))
            for l in range(lmax + 1)
            for p in [-1, 1]
        ])
        irreps_node_out = o3.Irreps([(energy_levels, (0, 1))])

        self.mp_edges = MessagePassing(
            irreps_node_sequence=[irreps_node_input] + mpn_layers * [irreps_node_hidden_edges] + [irreps_node_hidden_edges],
            irreps_node_attr=irreps_node_attr,
            irreps_edge_attr=irreps_edge_attr,
            fc_neurons=[number_of_basis, mul, 2*mul],
            num_neighbors=num_neighbors
        )

        mp_final_input = self.mp_edges.irreps_node_output
        self.mp_final = MessagePassing(
            irreps_node_sequence=[mp_final_input] + [irreps_node_hidden_edges] + [irreps_node_out],
            irreps_node_attr=irreps_node_attr,
            irreps_edge_attr=irreps_edge_attr,
            fc_neurons=[number_of_basis, mul, 2*mul],
            num_neighbors=num_neighbors
        )
        self.irreps_mp_output = self.mp_final.irreps_node_output
        self.num_neighbors = num_neighbors
    
    def forward(self, embed_on_edges, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding, batch) -> torch.Tensor:
        x = self.mp_edges(embed_on_edges, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding)
        atoms_fec = self.mp_final(x, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding)
        x = scatter(atoms_fec, batch, dim=0)
        print('x', x)
        return F.gumbel_softmax(x, tau=100, dim=1), atoms_fec


class Generator(nn.Module):
    def __init__(
        self,
        irreps_node_input: o3.Irreps,
        irreps_node_attr: o3.Irreps,
        irreps_edge_attr: o3.Irreps,
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
        irreps_node_out = o3.Irreps([(1, (1, -1))])

        self.mp1 = MessagePassing(
            irreps_node_sequence=[irreps_node_input] + mpn_layers * [irreps_node_hidden] + [irreps_node_out],
            irreps_node_attr=irreps_node_attr,
            irreps_edge_attr=irreps_edge_attr,
            fc_neurons=[number_of_basis, mul, 2*mul],
            num_neighbors=num_neighbors,
        )
        self.irreps_mp_output = self.mp1.irreps_node_output


    
    def forward(self, x, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding, batch) -> torch.Tensor:
        return self.mp1(x, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding)


class GeometricNet(LightningModule):
    def __init__(
        self,
        sample: Data,
        lr: float,
        b1: float = 0.5,
        b2: float = 0.999,
        energy_levels: int = 40,
        lmax: int = 2,
        number_of_basis: int = 10,
        mpn_layers: int = 4,
        max_radius: float = 6.0,
        mul: int = 64,
        use_forces: bool = False,
        **kwargs
    ):
        super(GeometricNet, self).__init__()
        self.save_hyperparameters('lr', 'b1', 'b2')

        self.name = 'geometric'
        self.test_criterion_initials = 'acc'
        self.fe_loss = smp.utils.losses.DiceLoss()
        self.use_forces = use_forces
        data_edges, data_bonds = sample
        self.num_nodes = data_edges.num_nodes / len(data_edges.e_label)

        irreps_node_input = f'{data_edges.x.size(1)}x0e+1x1o' if use_forces else o3.Irreps(f'{mul}x0e')
        irreps_node_attr = o3.Irreps(f'{data_edges.node_attr.size(1)}x0e')
        irreps_edge_attr = o3.Irreps('0e') + o3.Irreps.spherical_harmonics(lmax)
        irreps_embed_out = o3.Irreps(f'{mul//2}x0e+{mul//2}x0o+{mul//4}x1e+{mul//4}x1o+{mul//8}x2e+{mul//8}x2o')

        self.fc = FullyConnectedNet([data_edges.x.size(1), mul, mul], F.gelu) if use_forces else None
        
        self.embedding = Embedding(
            irreps_node_input=irreps_node_input,
            irreps_node_attr=irreps_node_attr,
            irreps_edge_attr=irreps_edge_attr,
            irreps_node_out=irreps_embed_out,
            lmax=lmax,
            mul=mul,
            mpn_layers=mpn_layers,
            max_radius=max_radius,
            number_of_basis=number_of_basis,
            num_neighbors=self.num_nodes,
            use_forces=use_forces
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
            num_neighbors=self.num_nodes
        )

        atoms_fec_irreps = self.energy_predictor.irreps_mp_output
        # self.generator = Generator(
        #     irreps_node_input=atoms_fec_irreps,
        #     irreps_node_attr=irreps_node_attr,
        #     irreps_edge_attr=irreps_edge_attr,
        #     lmax=lmax,
        #     mul=mul,
        #     mpn_layers=mpn_layers,
        #     number_of_basis=number_of_basis,
        #     num_neighbors=self.num_nodes
        # )

    def forward(self, data_edges: Union[Data, Dict[str, torch.Tensor]], data_bonds: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if self.use_forces:
            data_edges['x'] = self.fc(data_edges['x'])
        embed_on_edges, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding, batch = self.embedding(data_edges, data_bonds)
        free_energy, atoms_fec = self.energy_predictor(embed_on_edges, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding, batch)
        # edge_attr = torch.ones_like(edge_attr)
        # edge_length_embedding = torch.ones_like(edge_length_embedding)
        # new_pos = self.generator(atoms_fec, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding, batch)
        return free_energy, None # new_pos
    
    def free_energy_loss(self, y_hat, y):
        return self.fe_loss(y_hat, y)
    
    def autoencoder_loss(self, y_hat, y):
        return F.mse_loss(y_hat, y)
    
    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_e = torch.optim.Adam(list(self.embedding.parameters()) + list(self.energy_predictor.parameters()), lr=lr, betas=(b1, b2))
        # opt_g = torch.optim.Adam(list(self.embedding.parameters()) + list(self.generator.parameters()), lr=lr, betas=(b1, b2))
        return [opt_e], []
    
    def training_step(self, data, data_idx): # optimizer_idx <- add param if train with multiple optimizers
        data_edges, data_bonds = data
        embed_on_edges, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding, batch = self.embedding(data_edges, data_bonds)
        # train energy predictor
        # if optimizer_idx == 0: <- use if multiple optimizers
        y_hat, atoms_fec = self.energy_predictor(embed_on_edges, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding, batch)
        print('fe', y_hat)
        e_loss = self.free_energy_loss(y_hat.squeeze(-1), data_edges.e_label)
        tqdm_dict = {"e_loss": e_loss.item()}
        self.log_dict(tqdm_dict)
        output = OrderedDict({"loss": e_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
        return output
        
        # train dihedrals predictor
        # if optimizer_idx == 1:
        #     with torch.no_grad():
        #         _, atoms_fec = self.energy_predictor(x, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding, batch)
        #     edge_attr = torch.ones_like(edge_attr)
        #     edge_length_embedding = torch.ones_like(edge_length_embedding)
        #     y_hat = self.generator(atoms_fec, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding, batch)
        #     g_loss = self.autoencoder_loss(y_hat.squeeze(-1), data.pos)
        #     tqdm_dict = {"g_loss": g_loss.item()}
        #     self.log_dict(tqdm_dict)
        #     output = OrderedDict({"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
        #     return output
    
    def test_step(self, data, data_idx):
        data_edges, data_bonds = data
        energy_predictions, energy_targets = [], []
        graph_indexes = []

        free_energy, new_pos = self(data_edges, data_bonds)
        for i in range(free_energy.size(0)):
            energy_predictions.append(torch.argmax(free_energy.squeeze(-1)[i]))
            energy_targets.append(torch.argmax(data_edges.e_label[i]))

            graph_indexes.append(data_edges.graph_index[i])
        
        test_loss = self.free_energy_loss(free_energy.squeeze(-1), data_edges.e_label)
        test_acc = torch.sum((torch.tensor(energy_predictions) == torch.tensor(energy_targets))) / (len(energy_targets) * 1.0)
        data_edges.pos = new_pos
        return {
            'test_acc': test_acc,
            'test_loss': test_loss,
            'energy_predictions': energy_predictions,
            'energy_targets': energy_targets,
            'graph_indexes': graph_indexes,
            'new_molecule': data_edges
        }
