import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch


DATA_DIR = "data/ala_dipep"
LABELS_FILE = "phi-psi-free-energy.txt"
N_SAMPLES = 30000


def visualize3D(all_points, all_mean_forces_length, all_mean_forces_versor, all_var_forces, all_fes, elements, data_dir):
    import plotly.graph_objects as go

    fig = plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    data = []
    
    for atom_idx in range(len(all_mean_forces_versor[0])):
        x_forces = []
        y_forces = []
        z_forces = []
        
        for i in range(len(all_points)):
            x_forces.extend([all_points[i][0].item(), all_points[i][0].item() + all_mean_forces_versor[i][atom_idx, 0].item(), None])
            y_forces.extend([all_points[i][1].item(), all_points[i][1].item() + all_mean_forces_versor[i][atom_idx, 1].item(), None])
            z_forces.extend([all_fes[i]/10.0, all_fes[i]/10.0 + all_mean_forces_versor[i][atom_idx, 2].item(), None])
        
        trace_forces = go.Scatter3d(
        x=x_forces,
        y=y_forces,
        z=z_forces,
        name=f'forces_{elements[atom_idx]}',
        mode='lines',
        line=dict(color='brown', width=2),
        hoverinfo='none')

        data.append(trace_forces)
    
    for atom_idx in range(len(all_mean_forces_versor[0])):
        trace = go.Scatter3d(
        x=[x[0].item() for x in all_points],
        y=[y[1].item() for y in all_points],
        z=[z/10.0 for z in all_fes],
        name=f'atoms_{elements[atom_idx]}',
        mode='markers',
        text=[c[atom_idx].item() for c in all_var_forces],
        marker=dict(symbol='circle',
                size=5,
                color=[c[atom_idx].item() for c in all_var_forces])
        )

        data.append(trace)
    fig = go.Figure(data=data, layout=go.Layout(title=f'Forces'))
    
    # plt.show()
    fig.write_html(os.path.join(data_dir, f'forces_analysis-3D.html'))


def main(data_dir, labels_file, points, radius):
    if points is None:
        phi_t = torch.arange(-np.pi, np.pi, step=0.2)
        psi_t =torch.arange(-np.pi, np.pi, step=0.2)
        points = []
        for x in phi_t:
            for y in psi_t:
                points.append([x, y])
    
    with open(os.path.join(data_dir, labels_file), "r") as t:
        tensor = torch.as_tensor([[idx, float(phi), float(psi), float(f)] for idx, _, phi, psi, f in [[index] + x.split() for index, x in enumerate(t.readlines()) if index < N_SAMPLES]])
    
    all_points = []
    all_mean_forces_length = []
    all_mean_forces_versor = []
    all_var_forces = []
    all_fes = []
    for point in points:
        point = torch.Tensor(point)
        distances_from_point = tensor[:, 1:3].sub(point)
        distances_from_point = torch.norm(distances_from_point, dim=1)
        filtered_tensor = tensor[distances_from_point < radius]
        batch_forces = None
        for graph_index in filtered_tensor[:, 0]:
            forces_dir = os.path.join(data_dir, 'forces')
            if os.path.exists(forces_dir):
                df = pd.read_csv(os.path.join(forces_dir, f'{int(graph_index)}.csv'), header=0, quotechar='"')
                forces = torch.Tensor(df[['Fx', 'Fy', 'Fz']].values)
                if batch_forces is None:
                    batch_forces = forces.unsqueeze(-1)
                else:
                    batch_forces = torch.cat([batch_forces, forces.unsqueeze(-1)], dim=2)
        if batch_forces is not None:
            mean_forces = torch.mean(batch_forces, dim=-1)
            mean_forces_length = torch.norm(mean_forces, dim=-1)
            divider = torch.add(mean_forces_length, torch.zeros(3, 1)).T
            mean_forces_versor = mean_forces.div(divider)
            var_forces = torch.norm(torch.var(batch_forces, dim=1), dim=1)
            all_points.append(point)
            all_mean_forces_length.append(mean_forces_length)
            all_mean_forces_versor.append(mean_forces_versor)
            all_var_forces.append(var_forces)
            all_fes.append(torch.mean(filtered_tensor[:, 3]).item())
        else:
            all_points.append(point)
            all_mean_forces_length.append(torch.zeros((22, 1)))
            all_mean_forces_versor.append(torch.zeros((22, 3)))
            all_var_forces.append(torch.zeros((22, 1)))
            all_fes.append(0.0)
    
    elements = ['HH31','CH3','HH32','HH33','C','O','N','H','CA','HA','CB','HB1','HB2','HB3','C','O','N','H','CH3','HH31','HH32','HH33']
    visualize3D(all_points, all_mean_forces_length, all_mean_forces_versor, all_var_forces, all_fes, elements, data_dir=data_dir)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot mean forces in the neighbourhood of a point in the FES')
    parser.add_argument('--point', default=None, help='Point on the FES where to compute forces mean and variance (in radiants). Default: 0,0')
    parser.add_argument('--radius', default=0.1, help='Point on the FES where to compute forces mean and variance (in radiants). Default: 0,0')

    args = parser.parse_args()
    point = [torch.Tensor([float(x) for x in args.point.replace('~', '-').split(',')])] if args.point else None
    main(DATA_DIR, LABELS_FILE, points=point, radius=float(args.radius))