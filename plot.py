import argparse
import json
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import glob


DATA_DIR = f"/scratch/carrad/free-energy-gnn/ala_dipep_old"
N_SAMPLES = 50000

def plot(f: str):
    with open(f, "r") as l:
        inference = json.load(l)
    
    graph_samples = []
    for i in range(N_SAMPLES):
        with open(f"{DATA_DIR}/{i}-dihedrals-graph-nosin.pickle", "rb") as p:
            debruijn = pickle.load(p)

        graph_samples.append(debruijn)
    
    phi_list = [-180, 180]
    psi_list = [-180, 180]
    all_e = inference['energy_predicted'] + inference['energy_target']
    min_e, max_e = min(all_e), max(all_e)
    energy_list = [min_e, max_e]
    target_energy_list = [min_e, max_e]
    for i, (energy, target_energy, graph_index) in enumerate(zip(inference['energy_predicted'], inference['energy_target'], inference['test_frames'])):
        graph = graph_samples[graph_index]
        phi = graph[0][0, 0]
        psi = graph[0][1, 0]
        phi_list.append(phi)
        psi_list.append(psi)
        energy_list.append(energy)
        target_energy_list.append(target_energy)
    error_energy_list = list(np.abs(np.array(energy_list) - np.array(target_energy_list)))
    error_energy_list[:2] = [0, max_e - min_e]
    
    fig, axs = plt.subplots(3, figsize=(8, 12))
    axs[0].scatter(psi_list, phi_list, s=1, c=energy_list, cmap='turbo')
    axs[1].scatter(psi_list, phi_list, s=1, c=error_energy_list, cmap='turbo')
    axs[2].scatter(psi_list, phi_list, s=1, c=target_energy_list, cmap='turbo')
    dirname = '/'.join(f.split("/")[:-1])
    fig.savefig(f'{dirname}/plot.png', dpi=fig.dpi)
    del fig

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FES computation of Alanine Dipeptide using Geometric Deep Learning')
    parser.add_argument('--dir', required=False, help='Directory containing the results to be plotted')

    args = parser.parse_args()
    for f in glob.glob(os.path.join(args.dir, '**/result.json'), recursive=True):
        plot(f)