import argparse
import glob
import json
import os
from biopandas.pdb.pandas_pdb import PandasPdb
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import torch
from dataset import create_transform
from torch_geometric.data.data import Data


DATA_DIR = "data/ala_dipep"
LABELS_FILE = "phi-psi-free-energy.txt"
BONDS_FILE = "ala_dipep_bonds.csv"
PARTIAL_CHARGES_FILE = "ala_dipep_partial_charges.csv"
N_SAMPLES = 50000

def animate3D(df_p, df_t, data, color, data_dir, prediction_file):
    frames = []

    for k, d in enumerate(data):
        x_bonds = []
        y_bonds = []
        z_bonds = []
        
        for i in range(d.bonds.size(1)):
            x_bonds.extend([d.pos[d.bonds[0][i], 0].item(), d.pos[d.bonds[1][i], 0].item(), None])
            y_bonds.extend([d.pos[d.bonds[0][i], 1].item(), d.pos[d.bonds[1][i], 1].item(), None])
            z_bonds.extend([d.pos[d.bonds[0][i], 2].item(), d.pos[d.bonds[1][i], 2].item(), None])
        
        trace_bonds = go.Scatter3d(
        x=x_bonds,
        y=y_bonds,
        z=z_bonds,
        name='bonds',
        mode='lines',
        line=dict(color='red', width=6),
        hoverinfo='none')

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

        traces = [trace_bonds, trace_nodes]
        frames.append(
            go.Frame(
            data=traces,
            name=f'p{k}',
            layout=go.Layout(title=f'Phi: {d.dihedrals[0, 0].item() / np.pi * 180} - Psi: {d.dihedrals[0, 1].item() / np.pi * 180} - Free Energy: {torch.argmax(d.e_label).item()}'),
            traces=[0, 1]
            )
        )

        frames.append(
            go.Frame(
                    data=[
                        go.Scatter3d(
                            x=df_t.phi[::200], 
                            y=df_t.psi[::200],
                            z=df_t.fe[::200] / 5,
                            marker=dict(color=df_t.fe[::200], size=2),
                            name='target',
                            mode='markers',
                            text=df_t.graph_index[::200],
                        ),
                        go.Scatter3d(
                            x=df_t.phi[[df_t.graph_index[k]]], 
                            y=df_t.psi[[df_t.graph_index[k]]],
                            z=df_t.fe[[df_t.graph_index[k]]] / 5,
                            marker=dict(color='red', size=5),
                            name='mol',
                            mode='markers',
                            text=df_t.graph_index[[k]],
                        )
                        ],
                    name=f'q{k}',
                    layout=go.Layout(title=f'q{k}'),
                    traces=[2, 3]
                    )
        )
    
    # Create and add slider
    steps_pred = [dict(method='animate',
                  args=[[f'p{k}', f'q{k}'],
                        dict(mode='immediate',
                             frame=dict(duration=1),
                             transition=dict(duration=0)
                             )
                        ],
                  label=f'p{k}'
                  ) for k in range(0, len(data))]

    sliders = [
        dict(
            x=0.1,
            y=0,
            len=0.9,
            pad=dict(b=10, t=50),
            active=0,
            steps=steps_pred,
            currentvalue=dict(font=dict(size=20), prefix="", visible=True, xanchor='right'),
            transition=dict(easing="cubic-in-out", duration=1)),
    ]

    fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'scene'}, {'type': 'scene'}]])

    fig.add_trace(trace=go.Scatter3d(x=[], y=[], z=[], mode="markers"), row=1, col=1)
    fig.add_trace(trace=go.Scatter3d(x=[], y=[], z=[], mode="lines"), row=1, col=1)
    fig.add_trace(trace=go.Scatter3d(x=[], y=[], z=[], mode="markers"), row=1, col=2)
    fig.add_trace(trace=go.Scatter3d(x=[], y=[], z=[], mode="lines"), row=1, col=2)
    
    fig.update(frames=frames)
    fig.update_layout(
            scene = dict(
                camera=dict(eye=dict(x=1.15, y=1.15, z=0.8)),
                xaxis=dict(range=[-4, 4]),
                yaxis=dict(range=[-4, 4]),
                zaxis=dict(range=[-4, 4]),
                aspectratio=dict(x=1, y=1, z=1)
            ),
            updatemenus=[dict(
                            type="buttons",
                            buttons=[dict(label="Play",
                            method="animate",
                            args=[None, dict(frame=dict(duration=1))])])], 
            sliders=sliders
            )

    fname = prediction_file.split('/')[-2] if prediction_file is not None else 'all'
    fig.write_html(os.path.join(data_dir, f'{fname}-mol_anim-3D.html'))

def main(data_dir, labels_file, bonds_file, partial_charges_file, prediction_files: list):
    data = []

    for prediction_file in prediction_files:
        if prediction_file is None:
            with open(os.path.join(data_dir, labels_file), "r") as t:
                target = np.array([[float(idx), float(phi), float(psi), float(fe)] for idx, phi, psi, fe in [x.split() for x in t.readlines()]])
                predicted = np.array([[float(idx), float(phi), float(psi), float(0.0)] for idx, phi, psi, fe in target])
        else:
            with open(prediction_file, "r") as l:
                inference = json.load(l)
                predicted = inference['energy_predicted']
                target = inference['energy_target']
        
        df_p = pd.DataFrame(predicted, columns=['graph_index', 'phi', 'psi', 'fe'])
        df_t = pd.DataFrame(target, columns=['graph_index', 'phi', 'psi', 'fe'])

        data = []
        for graph_index in df_p.graph_index.values[:200]:
            ppdb = PandasPdb().read_pdb(os.path.join(data_dir, 'pdb', f'{int(graph_index)}.pdb'))
            m = {'atoms': ppdb.df['ATOM'], 'graph_index': int(graph_index)}
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
            data.append(d)

        mul = torch.tensor([i for i in range(4)], dtype=torch.float32)
        animate3D(df_p, df_t, data, color=torch.sum((d.x[:, :4] != 0) * mul, dim=1), data_dir=os.path.join(data_dir, 'animations'), prediction_file=prediction_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FES computation of Alanine Dipeptide using Geometric Deep Learning')
    parser.add_argument('--dir', required=False, help='Directory containing the results to be plotted')

    args = parser.parse_args()
    prediction_files = glob.glob(os.path.join(args.dir, '**/result.json'), recursive=True) if args.dir else [None]
    main(DATA_DIR, LABELS_FILE, BONDS_FILE, PARTIAL_CHARGES_FILE, prediction_files=prediction_files)