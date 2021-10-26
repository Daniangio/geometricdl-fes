import ast
import json
import os
import pickle
from biopandas.pdb.pandas_pdb import PandasPdb
import torch

from torch_geometric.data import Data
import pandas as pd
import functools
import numpy as np
from torch_cluster import radius_graph

from torch.utils.data import Dataset as BaseDataset

def indexes_to_one_hot(indexes, n_dims=None):
    """Converts a vector of indexes to a batch of one-hot vectors. """
    indexes = indexes.type(torch.int64).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(indexes)) + 1
    one_hots = torch.zeros(indexes.size()[0], n_dims).scatter_(1, indexes, 1)
    one_hots = one_hots.view(*indexes.shape, -1)
    return one_hots


def real_number_batch_to_one_hot_vector_bins(real_numbers, bins):
    """Converts a batch of real numbers to a batch of one hot vectors for the bins the real numbers fall in."""
    _, indexes = (real_numbers.view(-1, 1) - bins.view(1, -1)).abs().min(dim=1)
    return indexes_to_one_hot(indexes, n_dims=bins.shape[0])

def real_number_to_binned_gaussian(real_numbers, bins):
    from scipy.stats import norm
    x = torch.add(torch.zeros_like(real_numbers), bins)
    p = torch.from_numpy(norm.pdf(x, real_numbers, 0.03))
    p = torch.div(p, torch.sum(p, dim=1).broadcast_to(p.size()[::-1]).T)
    p = torch.round(p * 10**3) / (10**3)
    return p.unsqueeze(1).float()


def create_transform(data_dir, labels_file, bonds_file, partial_charges_file, use_dihedrals: bool, normalize_labels: bool=True, energy_levels: int=20):
    # Retrieve atom bonds
    df = pd.read_csv(os.path.join(data_dir, bonds_file), header=0, quotechar='"', converters={'bonds':ast.literal_eval})
    src_atoms = df['index'].values
    trg_atoms = df['bonds'].values

    bonds = []
    for src_atom in src_atoms:
        for trg_atom in trg_atoms[src_atom-1]:
            bonds.append([src_atom-1, trg_atom-1])
    bonds = torch.Tensor(bonds).int().T

    # Retrieve amot partial charges
    partial_charges = pd.read_csv(os.path.join(data_dir, partial_charges_file), header=0)
    partial_charges = torch.Tensor(partial_charges.charge.values)

    # Retrieve labels
    with open(os.path.join(data_dir, labels_file), "r") as t:
        tensor = torch.as_tensor([[float(phi), float(psi), float(f)] for _, phi, psi, f in [x.split() for x in t.readlines()]])
        phi_psi = tensor[:, :2].unsqueeze(1)
        labels = tensor[:, 2]
        if normalize_labels:
            # labels_std = torch.std(labels, dim=0)
            # labels_mean = torch.mean(labels, dim=0)
            # labels = ((labels - labels_mean) / labels_std).reshape(shape=(len(labels), 1))
            labels = labels.view(labels.size(0), -1)
            labels -= labels.min(0, keepdim=True)[0]
            labels /= labels.max(0, keepdim=True)[0]
            # labels = real_number_batch_to_one_hot_vector_bins(labels, torch.linspace(0.0, 1.0, steps=10))
            labels = real_number_to_binned_gaussian(labels, torch.linspace(0.0, 1.0, steps=energy_levels))
    return functools.partial(prepare_transform, data_dir=data_dir, use_dihedrals=use_dihedrals, phi_psi=phi_psi, bonds=bonds, partial_charges=partial_charges, labels=labels)


def prepare_transform(item, data_dir: str, use_dihedrals: bool, phi_psi: torch.Tensor, bonds: torch.Tensor, partial_charges:torch.Tensor, labels: torch.Tensor):
    # t = time.time()
    dihedrals = phi_psi[item['graph_index']]
    e_label = labels[item['graph_index']]
    if use_dihedrals:
        with open(f'{data_dir}/{item["graph_index"]}-dihedrals-graph.pickle', "rb") as p:
            debruijn = pickle.load(p)
        debruijn_edge_attr, debruijn_edge_index, _ = debruijn
        node_embedding = torch.eye(len(debruijn_edge_attr))
        debruijn_node_input = node_embedding # torch.cat([debruijn_node_input, node_embedding], dim=1)
        #debruijn_edge_attr = expand_edge_attr(debruijn_edge_attr, debruijn_edge_index)
        return None, debruijn_edge_index, debruijn_node_input, None, debruijn_edge_attr, bonds, dihedrals, e_label, None
    element_mapping = { # Mapping che non differenzia gli stessi atomi che hanno bond diversi
        'HH31': 0,
        'HH32': 0,
        'HH33': 0,
        'HB1': 0,
        'HB2': 0,
        'HB3': 0,
        'CH3': 1,
        'C': 1,
        'O': 2,
        'N': 3,
        'H': 0,
        'CA': 1,
        'HA': 0,
        'CB': 1
    }
    # element_mapping = { # Mapping che considera diversi gli atomi piu importanti in base ai loro bond
    #     'HH31': 0,
    #     'HH32': 0,
    #     'HH33': 0,
    #     'HB1': 0,
    #     'HB2': 0,
    #     'HB3': 0,
    #     'CH3': 1,
    #     'C': 2,
    #     'O': 3,
    #     'N': 4,
    #     'H': 5,
    #     'CA': 6,
    #     'HA': 7,
    #     'CB': 8
    # }
    # element_mapping = {
    #     'C': 0,
    #     'O': 1,
    #     'N': 2,
    #     'H': 3,
    #     'CX': 4,
    #     'HC': 5,
    #     'CT': 6,
    #     'H1': 7
    # }
    if type(item['atoms']) != pd.DataFrame:
        item['atoms'] = pd.DataFrame(**item['atoms'])
    coords = item['atoms'][['x_coord', 'y_coord', 'z_coord']].values
    elements = item['atoms']['atom_name'].values
    # coords = item['atoms'][['x', 'y', 'z']].values
    # elements = item['atoms']['type'].values
    
    filter = np.array([i for i, e in enumerate(elements) if e in element_mapping])
    coords = coords[filter]
    elements = elements[filter]

    # Make one-hot
    elements_int = np.array([element_mapping[e] for e in elements])
    one_hot = np.zeros((elements.size, max(element_mapping.values()) + 1))
    one_hot[np.arange(elements.size), elements_int] = 1
    node_input = torch.tensor(one_hot, dtype=torch.float32)

    # Atom coordinates
    pos = torch.tensor(coords, dtype=torch.float32)

    # Build edges
    max_radius = 10.0
    edge_index = radius_graph(pos, max_radius, max_num_neighbors=elements.size)

    # apply random rotation
    # pos = torch.einsum('zij,zaj->zai', o3.rand_matrix(len(pos)), pos)
    one_hot[np.arange(elements.size), elements_int] = partial_charges
    node_attr = torch.tensor(one_hot, dtype=torch.float32)
    edge_attr = torch.ones(edge_index.shape[1], 1, dtype=torch.float32)

    # print(time.time() - t)
    return pos, edge_index, node_input, node_attr, edge_attr, bonds, dihedrals, e_label, elements


def expand_edge_attr(attr, graph):
    attr_d = {}
    expanded_attr = torch.zeros((graph.shape[1], attr.shape[1]))
    idx = 0
    for i, (from_, to_) in enumerate(zip(graph[0], graph[1])):
        if (to_.item(), from_.item()) not in attr_d.keys():
            expanded_attr[i] = attr[idx]
            attr_d[(from_.item(), to_.item())] = attr[idx]
            idx += 1
        else:
            expanded_attr[i] = attr_d[(to_.item(), from_.item())]
    return expanded_attr

def make_graph_directed(graph):
    edge_index = np.zeros((len(graph), graph.shape[1] // 2))
    idx = 0
    g = graph.numpy().T
    for from_, to_ in zip(graph[0], graph[1]):
        if not any(np.equal(g[:idx, :], [to_, from_]).all(1)):
            edge_index[0, idx] = from_
            edge_index[1, idx] = to_
            idx += 1
            if idx == edge_index.shape[1]:
                break
    return torch.tensor(edge_index)


class Dataset(BaseDataset):
    def __init__(self, data_dir, indexes, transform, use_dihedrals):
        self.data_dir = os.path.join(data_dir, 'pdb') if not use_dihedrals else data_dir
        self.indexes = indexes
        self.transform = transform
        self.use_dihedrals = use_dihedrals

    def __getitem__(self, i):
        if self.use_dihedrals:
            try:
                with open("{}/{}-graph-df.pickle".format(self.data_dir, self.indexes[i]), "rb") as p:
                    m = {'atoms': pickle.load(p), 'graph_index': self.indexes[i]}
            except FileNotFoundError:
                with open("{}/{}.json".format(self.data_dir, self.indexes[i]), "r") as f:
                    raw = json.load(f)
                df = pd.DataFrame(raw['atoms'])
                df.to_pickle("{}/{}-graph-df.pickle".format(self.data_dir, self.indexes[i]))
                m = {'atoms': df, 'graph_index': self.indexes[i]}
        else:
            ppdb = PandasPdb().read_pdb(os.path.join(self.data_dir, f'{self.indexes[i]}.pdb'))
            m = {'atoms': ppdb.df['ATOM'], 'graph_index': self.indexes[i]}
        pos, edge_index, node_input, node_attr, edge_attr, bonds, dihedrals, e_label, elements = self.transform(m)
        d = Data(x=node_input,
                 pos=pos,
                 edge_index=edge_index,
                 node_attr=node_attr,
                 edge_attr=edge_attr,
                 bonds=bonds,
                 dihedrals=dihedrals,
                 e_label=e_label,
                 elements=elements,
                 graph_index=self.indexes[i])
        return d

    def __len__(self):
        return len(self.indexes)