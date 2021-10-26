import argparse
import glob
import json
import os
import pandas as pd
import pickle


DATA_DIR = "/scratch/carrad/free-energy-gnn/ala_dipep_old"
LABELS_FILE = "/scratch/carrad/free-energy-gnn/free-energy-old.dat"
N_SAMPLES = 50000

def animate3D(df, data_dir, prediction_file):
    import plotly.graph_objects as go

    frames = [go.Frame(
                    data=[
                        go.Scatter3d(
                            x=df.phi[:k+1], 
                            y=df.psi[:k+1],
                            z=df.predicted[:k+1],
                            marker=dict(color=df.predicted[:k+1], size=2),
                            name='predicted',
                            mode='markers',
                            text=df.graph_index[:k+1],
                        )],
                    name=f'p{k}',
                    layout=go.Layout(title=f'p{k}')
                    ) for k in range(0, len(df.phi), 40)]
    
    frames += [go.Frame(
                    data=[
                        go.Scatter3d(
                            x=df.phi[:k+1], 
                            y=df.psi[:k+1],
                            z=df.target[:k+1],
                            marker=dict(color=df.target[:k+1], size=2),
                            name='target',
                            mode='markers',
                            text=df.graph_index[:k+1],
                        )],
                    name=f't{k}',
                    layout=go.Layout(title=f't{k}')
                    ) for k in range(0, len(df.phi), 40)]
    
    # Create and add slider
    steps_pred = [dict(method='animate',
                  args=[["p{}".format(k)],
                        dict(mode='immediate',
                             frame=dict(duration=1),
                             transition=dict(duration=0)
                             )
                        ],
                  label=f'p{k}'
                  ) for k in range(0, len(df.phi), 40)]
    steps_target = [dict(method='animate',
                  args=[["t{}".format(k)],
                        dict(mode='immediate',
                             frame=dict(duration=1),
                             transition=dict(duration=0)
                             )
                        ],
                  label=f't{k}'
                  ) for k in range(0, len(df.phi), 40)]

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
        dict(
            x=0.1,
            y=-0.2,
            len=0.9,
            pad=dict(b=10, t=50),
            active=0,
            steps=steps_target,
            currentvalue=dict(font=dict(size=20), prefix="", visible=True, xanchor='right'),
            transition=dict(easing="cubic-in-out", duration=1))
    ]
        
    fig = go.Figure(
            data=[go.Scatter3d(x=[], y=[], z=[], mode="markers")],
            frames=frames,
            layout=go.Layout(
            scene = dict(
            xaxis=dict(range=[-180, 180], autorange=False),
            yaxis=dict(range=[-180, 180], autorange=False),
            zaxis=dict(range=[min(df.target), max(df.target)], autorange=False),
            ),
            sliders=sliders,
            updatemenus=[dict(
                            type="buttons",
                            buttons=[dict(label="Play",
                            method="animate",
                            args=[None, dict(frame=dict(duration=1))])])],
            ))

    
    fig.update(frames=frames)

    fname = prediction_file.split('/')[-2] if prediction_file is not None else 'all'
    fig.write_html(os.path.join(data_dir, f'{fname}-animation-3D.html'))

def main(data_dir, labels_file, prediction_files: list):
    data = []

    for prediction_file in prediction_files:
        if prediction_file is None:
            with open(labels_file, "r") as t:
                target = [float(v) for v in t.readlines()]
                predicted = [0.0 for _ in target]
        else:
            with open(prediction_file, "r") as l:
                inference = json.load(l)
                predicted = inference['energy_predicted']
                target = inference['energy_target']
        
        graph_indexes = range(0, 50000, 10) if prediction_file is None else inference['test_frames']
        for i, graph_index in enumerate(graph_indexes):
            if i % 5 > 0:
                continue
            with open("{}/{}-dihedrals-graph-nosin.pickle".format(data_dir, graph_index), "rb") as p:
                debruijn = pickle.load(p)
                phi = debruijn[0][1].item()
                psi = debruijn[0][0].item()
            data.append([phi, psi, predicted[i], target[i], graph_index])
        
        df = pd.DataFrame(data, columns=['phi', 'psi', 'predicted', 'target', 'graph_index'])    
        animate3D(df, data_dir=os.path.join('data', 'ala_dipep', 'animations'), prediction_file=prediction_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FES computation of Alanine Dipeptide using Geometric Deep Learning')
    parser.add_argument('--dir', required=False, help='Directory containing the results to be plotted')

    args = parser.parse_args()
    prediction_files = glob.glob(os.path.join(args.dir, '**/result.json'), recursive=True) if args.dir else [None]
    main(DATA_DIR, LABELS_FILE, prediction_files=prediction_files)