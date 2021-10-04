import json
import pickle
import torch

from torch_geometric.data import Data
import pandas as pd
import functools
import numpy as np
from torch_cluster import radius_graph

from torch.utils.data import Dataset as BaseDataset

def create_transform(data_dir, labels_file, use_dihedrals: bool, normalize_labels: bool=True):
    with open(labels_file, "r") as t:
        labels = torch.as_tensor([[float(v)] for v in t.readlines()])
        if normalize_labels:
            labels_std = torch.std(labels, dim=0)
            labels_mean = torch.mean(labels, dim=0)
            labels = ((labels - labels_mean) / labels_std).reshape(shape=(len(labels), 1))
    return functools.partial(prepare_transform, data_dir=data_dir, use_dihedrals=use_dihedrals, labels=labels)


def prepare_transform(item, data_dir: str, use_dihedrals: bool, labels: torch.Tensor=None):
    # t = time.time()
    label = labels[item['graph_index']] if labels is not None else None

    if use_dihedrals:
        with open(f'{data_dir}/{item["graph_index"]}-dihedrals-graph.pickle', "rb") as p:
            debruijn = pickle.load(p)
        debruijn_node_input, debruijn_edge_index, _ = debruijn
        return None, debruijn_edge_index, debruijn_node_input, None, None, label
    element_mapping = {
        'C': 0,
        'O': 1,
        'N': 2,
        'H': 3,
        'CX': 4
        # 'HC': 5,
        # 'CT': 6,
        # 'H1': 7
    }
    if type(item['atoms']) != pd.DataFrame:
        item['atoms'] = pd.DataFrame(**item['atoms'])
    coords = item['atoms'][['x', 'y', 'z']].values
    elements = item['atoms']['type'].values
    
    filter = np.array([i for i, e in enumerate(elements) if e in element_mapping])
    coords = coords[filter]
    elements = elements[filter]

    # Make one-hot
    elements_int = np.array([element_mapping[e] for e in elements])
    one_hot = np.zeros((elements.size, len(element_mapping)))
    one_hot[np.arange(elements.size), elements_int] = 1

    pos = torch.tensor(coords, dtype=torch.float32)

    # Build edges
    max_radius = 10.0
    edge_index = radius_graph(pos, max_radius, max_num_neighbors=10)

    # apply random rotation
    # pos = torch.einsum('zij,zaj->zai', o3.rand_matrix(len(pos)), pos)
    node_input = torch.tensor(one_hot, dtype=torch.float32)
    node_attr = torch.tensor([], dtype=torch.float32)
    edge_attr = torch.tensor([], dtype=torch.float32)

    # print(time.time() - t)
    return pos, edge_index, node_input, node_attr, edge_attr, label

class Dataset(BaseDataset):
    def __init__(self, data_dir, indexes, transform):
        self.data_dir = data_dir
        self.indexes = indexes
        self.transform = transform

    def __getitem__(self, i): 
        try:
            with open("{}/{}-graph-df.pickle".format(self.data_dir, self.indexes[i]), "rb") as p:
                m = {'atoms': pickle.load(p), 'graph_index': self.indexes[i]}
        except FileNotFoundError:
            with open("{}/{}.json".format(self.data_dir, self.indexes[i]), "r") as f:
                raw = json.load(f)
            df = pd.DataFrame(raw['atoms'])
            df.to_pickle("{}/{}-graph-df.pickle".format(self.data_dir, self.indexes[i]))
            m = {'atoms': df, 'graph_index': self.indexes[i]}
        pos, edge_index, node_input, node_attr, edge_attr, label = self.transform(m)
        d = Data(x=node_input,
                 pos=pos,
                 edge_index=edge_index,
                 node_attr=node_attr,
                 edge_attr=edge_attr,
                 label=label,
                 graph_index=self.indexes[i])
        return d

    def __len__(self):
        return len(self.indexes)