import argparse
import os
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data.data import Data
from torch_geometric.utils import to_networkx
from biopandas.pdb import PandasPdb
from dataset import create_transform
from models.geometric import GeometricNet
from models.geometricphipsi import GeometricPhiPsiNet


DATA_DIR = "data/ala_dipep"
LABELS_FILE = "phi-psi-free-energy.txt"
BONDS_FILE = "ala_dipep_bonds.csv"
PARTIAL_CHARGES_FILE = "ala_dipep_partial_charges.csv"
N_SAMPLES = 50000

def visualize2D(G, graph_index, color, epoch=None, loss=None, data_dir=None):
    fig = plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])

    if torch.is_tensor(G):
        h = G.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
        if epoch is not None and loss is not None:
            plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    else:
        nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=True,
                         node_color=color, cmap="Set2")
    
    # plt.show()
    fig.savefig(os.path.join(data_dir, f'{graph_index}-2D.png'), dpi=fig.dpi)

def visualize3D(d, graph_index, color, edge_weights, data_dir):
    import plotly.graph_objects as go

    fig = plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])

    x_bonds = []
    y_bonds = []
    z_bonds = []
    
    for i in range(d.bonds.size(1)):
        x_bonds.extend([d.pos[d.bonds[0][i], 0].item(), d.pos[d.bonds[1][i], 0].item(), None])
        y_bonds.extend([d.pos[d.bonds[0][i], 1].item(), d.pos[d.bonds[1][i], 1].item(), None])
        z_bonds.extend([d.pos[d.bonds[0][i], 2].item(), d.pos[d.bonds[1][i], 2].item(), None])

    x_edges = []
    y_edges = []
    z_edges = []
    
    for i in range(d.edge_index.size(1)):
        x_edges.extend([d.pos[d.edge_index[0][i], 0].item(), d.pos[d.edge_index[1][i], 0].item(), None])
        y_edges.extend([d.pos[d.edge_index[0][i], 1].item(), d.pos[d.edge_index[1][i], 1].item(), None])
        z_edges.extend([d.pos[d.edge_index[0][i], 2].item(), d.pos[d.edge_index[1][i], 2].item(), None])
    
    x_forces = []
    y_forces = []
    z_forces = []
    
    for i in range(d.x.size(0)):
        x_forces.extend([d.pos[i, 0].item(), d.pos[i, 0].item() + d.forces[i, 0].item()/10.0, None])
        y_forces.extend([d.pos[i, 1].item(), d.pos[i, 1].item() + d.forces[i, 1].item()/10.0, None])
        z_forces.extend([d.pos[i, 2].item(), d.pos[i, 2].item() + d.forces[i, 2].item()/10.0, None])

    # create a trace for the bonds of the graph
    trace_bonds = go.Scatter3d(
        x=x_bonds,
        y=y_bonds,
        z=z_bonds,
        name='bonds',
        mode='lines',
        line=dict(color='red', width=6),
        hoverinfo='none')

    edge_src = d.edge_index[0]
    edge_dst = d.edge_index[1]
    edge_vec = d.pos[edge_src] - d.pos[edge_dst]
    edge_length = edge_vec.norm(dim=1)
    # create a trace for the edges of the graph
    trace_edges = go.Scatter3d(
        x=x_edges,
        y=y_edges,
        z=z_edges,
        name='graph_edges',
        mode='lines',
        line=dict(color='black', width=2),
        hoverinfo='none')
    
    # create a trace for the forces on aoms
    trace_forces = go.Scatter3d(
        x=x_forces,
        y=y_forces,
        z=z_forces,
        name='forces',
        mode='lines',
        line=dict(color='brown', width=2),
        hoverinfo='none')

    # create a trace for the nodes of the graph, which are atoms and are displayed using their x,y,z coordinates
    trace_nodes = go.Scatter3d(
        x=d.pos[:, 0],
        y=d.pos[:, 1],
        z=d.pos[:, 2],
        name='atoms',
        mode='markers',
        text=d.elements,
        marker=dict(symbol='circle',
                size=10,
                color=color)
        )

    #Include the traces we want to plot and create a figure
    data = [trace_edges, trace_nodes, trace_bonds, trace_forces]
    fig = go.Figure(data=data, layout=go.Layout(title=f'Phi: {d.dihedrals[0, 0].item() / np.pi * 180} - Psi: {d.dihedrals[0, 1].item() / np.pi * 180}'))
    
    # plt.show()
    fig.write_html(os.path.join(data_dir, f'{torch.argmax(d.e_label, dim=1).item()}-{graph_index}-3D.html'))

def visualize_from_pdb(data_dir, labels_file, bonds_file, partial_charges_file, graph_index):
    # with open("{}/{}-graph-df.pickle".format(DATA_DIR, graph_index), "rb") as p:
    #     m = {'atoms': pickle.load(p), 'graph_index': graph_index}
    ppdb = PandasPdb().read_pdb(os.path.join(data_dir, 'pdb', f'{graph_index}.pdb'))
    # print(ppdb.df['ATOM'])
    m = {'atoms': ppdb.df['ATOM'], 'graph_index': graph_index}
    transform = create_transform(data_dir, labels_file, bonds_file, partial_charges_file, use_dihedrals=False)
    pos, edge_index, node_input, node_attr, edge_attr, bonds, bonds_attr, dihedrals, forces, e_label, elements = transform(m)
    d = Data(
        x=node_input,
        pos=pos,
        edge_index=edge_index,
        node_attr=node_attr,
        edge_attr=edge_attr,
        bonds=bonds,
        bonds_attr=bonds_attr,
        forces=forces,
        e_label=e_label,
        dihedrals=dihedrals,
        elements=elements,
        graph_index=[graph_index])
    
    visualize(data_dir, graph_index, d)
    return d

def visualize(data_dir, graph_index, d):
    G = to_networkx(d, to_undirected=True)
    mul = torch.tensor([i for i in range(4)], dtype=torch.float32)
    # visualize2D(G, graph_index=graph_index, color=torch.sum(d.x * mul, dim=1), data_dir=os.path.join(data_dir, 'graphs'))
    edge_src, edge_dst = d['edge_index']
    edge_distances = d['pos'][edge_src] - d['pos'][edge_dst]
    edge_weights = edge_distances.norm(dim=1)
    visualize3D(d, graph_index=graph_index, color=torch.sum((d.x[:, :4] != 0) * mul, dim=1), edge_weights=edge_weights, data_dir=os.path.join(data_dir, 'graphs'))

'''
Creates an html file that shows the 3D structure of a molecule in an interactive way
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot the 2D graph of the molecule, with atoms as nodes and atom distances as edges. Plot also the 3D structure of the molecule')
    parser.add_argument('--graph', default=0, help='Index of the graph to be plotted. Default is 0')
    parser.add_argument('--weights', default=None, help='file containing the model weights to be loaded before to run inference, build and visualize a graph')
    parser.add_argument('--model', default='geometric', help='Name of the model to use')

    models = {
        'geometric': GeometricNet,
        'geometricphipsi': GeometricPhiPsiNet
    }

    args = parser.parse_args()
    data = visualize_from_pdb(DATA_DIR, LABELS_FILE, BONDS_FILE, PARTIAL_CHARGES_FILE, int(args.graph))
    
    if args.weights:
        model = models[args.model](data, 0)
        try:
            model.load_state_dict(torch.load(args.weights), strict=True)
            print(f'Model weights {args.weights} loaded')
        except Exception as e:
            print(f'Model weights could not be loaded: {str(e)}')
    
        model = model.cuda()
        model.eval()
        with torch.no_grad():
            data = data.cuda()
            results = model.test_step(data, int(args.graph))
            visualize(DATA_DIR, graph_index=f'G{args.graph}', d=results['new_molecule'].cpu())