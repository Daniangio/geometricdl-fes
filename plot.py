import json
import pickle
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from networkx.drawing.nx_pydot import write_dot
import matplotlib.pyplot as plt


def save_graph(nodes, edges, filename, to_undirected=True):
    graph = Data(x=nodes, edge_index=edges)
    nx_graph = to_networkx(graph, node_attrs=["x"], to_undirected=to_undirected)
    write_dot(nx_graph, filename)
    # dot_graph = to_pydot(nx_graph)
    # dot_graph.write(filename)
    pass

DATA_DIR = f"/scratch/carrad/free-energy-gnn/ala_dipep_old"
RESULTS_DATA_DIR = 'results/right'
RUN_NAME = 'linear-old-0929-1708-mae:0.00'
TARGET_FILE = f"/scratch/carrad/free-energy-gnn/free-energy-old.dat"
N_SAMPLES = 50000

if __name__ == '__main__':
    with open(f"{RESULTS_DATA_DIR}/{RUN_NAME}/result.json", "r") as l:
        inference = json.load(l)
    
    graph_samples = []
    for i in range(N_SAMPLES):
        with open("{}/{}-dihedrals-graph-nosin.pickle".format(DATA_DIR, i), "rb") as p:
            debruijn = pickle.load(p)

        graph_samples.append(debruijn)
    
    with open(TARGET_FILE, "r") as t:
        labels = torch.as_tensor([[float(v)] for v in t.readlines()])
        labels_std = torch.std(labels, dim=0)
        labels_mean = torch.mean(labels, dim=0)
        labels = ((labels - labels_mean) / labels_std).reshape(shape=(len(labels), 1))
    
    phi_list = [15, 20]
    psi_list = [15, 20]
    energy_list = [0, 1]
    target_energy_list = [0, 1]
    for i, (energy, graph_index) in enumerate(zip(inference['predicted'], inference['test_frames'])):
        if i % 3 != 0:
            continue
        graph = graph_samples[graph_index]
        phi = graph[0][0, 0]
        psi = graph[0][1, 0]
        phi_list.append(phi)
        psi_list.append(psi)
        energy_list.append(energy if energy > 0 else 0.0)
        target_energy_list.append(labels[graph_index][0].item())
    
    fig, axs = plt.subplots(2)
    axs[0].scatter(psi_list, phi_list, s=1, c=energy_list, cmap='turbo')
    axs[1].scatter(psi_list, phi_list, s=1, c=target_energy_list, cmap='turbo')
    fig.savefig(f'{RESULTS_DATA_DIR}/{RUN_NAME}/{RUN_NAME}.png', dpi=fig.dpi)