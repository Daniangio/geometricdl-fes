import argparse
import json
import os
import pickle
import matplotlib.pyplot as plt


DATA_DIR = f"/scratch/carrad/free-energy-gnn/ala_dipep_old"
N_SAMPLES = 50000

def plot(dir: str):
    with open(os.path.join(dir, "result.json"), "r") as l:
        inference = json.load(l)
    
    graph_samples = []
    for i in range(N_SAMPLES):
        with open(f"{DATA_DIR}/{i}-dihedrals-graph-nosin.pickle", "rb") as p:
            debruijn = pickle.load(p)

        graph_samples.append(debruijn)
    
    phi_list = [-180, 180]
    psi_list = [-180, 180]
    energy_list = [0, 9]
    target_energy_list = [0, 9]
    for i, (energy, target_energy, graph_index) in enumerate(zip(inference['predicted'], inference['target'], inference['test_frames'])):
        graph = graph_samples[graph_index]
        phi = graph[0][0, 0]
        psi = graph[0][1, 0]
        phi_list.append(phi)
        psi_list.append(psi)
        energy_list.append(energy)
        target_energy_list.append(target_energy)
    
    fig, axs = plt.subplots(2)
    axs[0].scatter(psi_list, phi_list, s=1, c=energy_list, cmap='turbo')
    axs[1].scatter(psi_list, phi_list, s=1, c=target_energy_list, cmap='turbo')
    filename = dir.split("/")
    filename = filename[-1] if len(filename[-1]) > 0 else filename[-2]
    fig.savefig(f'{dir}/{filename}.png', dpi=fig.dpi)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FES computation of Alanine Dipeptide using Geometric Deep Learning')
    parser.add_argument('--dir', required=True, help='Directory containing the results to be plotted')

    args = parser.parse_args()
    plot(args.dir)