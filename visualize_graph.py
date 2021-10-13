import argparse
import random
import torch
import os
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data.data import Data
from torch_geometric.utils import to_networkx

from dataset import create_transform


DATA_DIR = f"/scratch/carrad/free-energy-gnn/ala_dipep_old"
LABELS_FILE = "/scratch/carrad/free-energy-gnn/free-energy-old.dat"
N_SAMPLES = 50000

def visualize2D(G, graph_index, color, epoch=None, loss=None):
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
    fig.savefig(f'{graph_index}-2D.png', dpi=fig.dpi)

def visualize3D(G, graph_index, edge_weights):
    import plotly.graph_objects as go

    fig = plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])

    # plt.figure(figsize=(5,5))
    edges = G.edges()

    # ## update to 3d dimension
    spring_3D = nx.spring_layout(G, dim = 3, k = 0.5) # k regulates the distance between nodes
    # weights = [G[u][v]['weight'] for u,v in edges]
    # nx.draw(G, with_labels=True, node_color='skyblue', font_weight='bold',  width=weights, pos=pos)

    # we need to seperate the X,Y,Z coordinates for Plotly
    # NOTE: spring_3D is a dictionary where the keys are 1,...,6
    x_nodes= [spring_3D[key][0] for key in spring_3D.keys()] # x-coordinates of nodes
    y_nodes = [spring_3D[key][1] for key in spring_3D.keys()] # y-coordinates
    z_nodes = [spring_3D[key][2] for key in spring_3D.keys()] # z-coordinates

    #we need to create lists that contain the starting and ending coordinates of each edge.
    x_edges=[]
    y_edges=[]
    z_edges=[]

    #create lists holding midpoints that we will use to anchor text
    xtp = []
    ytp = []
    ztp = []

    #need to fill these with all of the coordinates
    for edge in edges:
        #format: [beginning,ending,None]
        x_coords = [spring_3D[edge[0]][0],spring_3D[edge[1]][0],None]
        x_edges += x_coords
        xtp.append(0.5*(spring_3D[edge[0]][0]+ spring_3D[edge[1]][0]))

        y_coords = [spring_3D[edge[0]][1],spring_3D[edge[1]][1],None]
        y_edges += y_coords
        ytp.append(0.5*(spring_3D[edge[0]][1]+ spring_3D[edge[1]][1]))

        z_coords = [spring_3D[edge[0]][2],spring_3D[edge[1]][2],None]
        z_edges += z_coords
        ztp.append(0.5*(spring_3D[edge[0]][2]+ spring_3D[edge[1]][2]))
    
    etext = [f'weight={w}' for w in edge_weights]

    trace_weights = go.Scatter3d(x=xtp, y=ytp, z=ztp,
        mode='markers',
        marker =dict(color='rgb(125,125,125)', size=1), #set the same color as for the edge lines
        text = etext, hoverinfo='text')

    #create a trace for the edges
    trace_edges = go.Scatter3d(
        x=x_edges,
        y=y_edges,
        z=z_edges,
        mode='lines',
        line=dict(color='black', width=2),
        hoverinfo='none')

    #create a trace for the nodes
    trace_nodes = go.Scatter3d(
        x=x_nodes,
        y=y_nodes,
        z=z_nodes,
        mode='markers',
        marker=dict(symbol='circle',
                size=10,
                color='skyblue')
        )

    #Include the traces we want to plot and create a figure
    data = [trace_edges, trace_nodes, trace_weights]
    fig = go.Figure(data=data)
    
    # plt.show()
    fig.write_html(f'{graph_index}-3D.html')

def main():
    graph_index = random.randint(0, N_SAMPLES)
    with open("{}/{}-graph-df.pickle".format(DATA_DIR, graph_index), "rb") as p:
        m = {'atoms': pickle.load(p), 'graph_index': graph_index}
    transform = create_transform(DATA_DIR, LABELS_FILE, use_dihedrals=False)
    pos, edge_index, node_input, node_attr, edge_attr, label = transform(m)
    d = Data(
        x=node_input,
        pos=pos,
        edge_index=edge_index,
        node_attr=node_attr,
        edge_attr=edge_attr,
        label=label,
        graph_index=graph_index)
    
    print(d)
    G = to_networkx(d, to_undirected=True)
    mul = torch.tensor([i for i in range(d.x.size(-1))], dtype=torch.float32)
    visualize2D(G, graph_index=graph_index, color=torch.sum(d.x * mul, dim=1))
    edge_src, edge_dst = d['edge_index']
    edge_distances = d['pos'][edge_src] - d['pos'][edge_dst]
    edge_weights = edge_distances.norm(dim=1)
    visualize3D(G, graph_index=graph_index, edge_weights=edge_weights)

if __name__ == '__main__':
    main()