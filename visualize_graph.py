import os
import random
import torch
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data.data import Data
from torch_geometric.utils import to_networkx
from biopandas.pdb import PandasPdb
from dataset import create_transform


DATA_DIR = "data/ala_dipep"
LABELS_FILE = "data/ala_dipep/phi-psi-free-energy.txt"
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


    #create lists holding midpoints that we will use to anchor text
    x_edges = []
    y_edges = []
    z_edges = []
    
    for i in range(d.edge_index.size(1)):
        x_edges.extend([d.pos[d.edge_index[0][i], 0].item(), d.pos[d.edge_index[1][i], 0].item(), None])
        y_edges.extend([d.pos[d.edge_index[0][i], 1].item(), d.pos[d.edge_index[1][i], 1].item(), None])
        z_edges.extend([d.pos[d.edge_index[0][i], 2].item(), d.pos[d.edge_index[1][i], 2].item(), None])

    # trace_weights = go.Scatter3d(x=xtp, y=ytp, z=ztp,
    #     mode='markers',
    #     marker =dict(color='rgb(125,125,125)', size=1), #set the same color as for the edge lines
    #     text = etext, hoverinfo='text')

    #create a trace for the edges of the graph
    trace_edges = go.Scatter3d(
        x=x_edges,
        y=y_edges,
        z=z_edges,
        name='graph_edges',
        mode='lines',
        line=dict(color='black', width=2),
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
    data = [trace_edges, trace_nodes]
    fig = go.Figure(data=data)
    
    # plt.show()
    fig.write_html(os.path.join(data_dir, f'{graph_index}-3D.html'))

def main(data_dir, labels_file):
    graph_index = random.randint(0, N_SAMPLES)
    # with open("{}/{}-graph-df.pickle".format(DATA_DIR, graph_index), "rb") as p:
    #     m = {'atoms': pickle.load(p), 'graph_index': graph_index}
    ppdb = PandasPdb().read_pdb(os.path.join(data_dir, 'pdb', f'{graph_index}.pdb'))
    print(ppdb.df['ATOM'])
    m = {'atoms': ppdb.df['ATOM'], 'graph_index': graph_index}
    transform = create_transform(data_dir, labels_file, use_dihedrals=False)
    pos, edge_index, node_input, node_attr, edge_attr, label, elements = transform(m)
    d = Data(
        x=node_input,
        pos=pos,
        edge_index=edge_index,
        node_attr=node_attr,
        edge_attr=edge_attr,
        label=label,
        elements=elements,
        graph_index=graph_index)
    
    G = to_networkx(d, to_undirected=True)
    mul = torch.tensor([i for i in range(d.x.size(-1))], dtype=torch.float32)
    visualize2D(G, graph_index=graph_index, color=torch.sum(d.x * mul, dim=1), data_dir=os.path.join(data_dir, 'graphs'))
    edge_src, edge_dst = d['edge_index']
    edge_distances = d['pos'][edge_src] - d['pos'][edge_dst]
    edge_weights = edge_distances.norm(dim=1)
    visualize3D(d, graph_index=graph_index, color=torch.sum(d.x * mul, dim=1), edge_weights=edge_weights, data_dir=os.path.join(data_dir, 'graphs'))

if __name__ == '__main__':
    main(DATA_DIR, LABELS_FILE)