import argparse
import json
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import glob

import torch


DATA_DIR = 'data/ala_dipep'
LABELS_FILE = 'phi-psi-free-energy.txt'
N_SAMPLES = 30000

def plot(dir: str, data_dir: str, labels_file: str):
    with open(os.path.join(data_dir, labels_file), "r") as t:
        phi_psi = torch.Tensor([[float(phi), float(psi)] for _, phi, psi, _ in [x.split() for x in t.readlines()]])
        phi_psi = phi_psi / np.pi * 180

    for result_file in glob.glob(os.path.join(dir, '**/result.json'), recursive=True):
        with open(result_file, "r") as f:
            inference = json.load(f)
        
        phi_list = [-180, 180]
        psi_list = [-180, 180]
        pred, target = inference.get('energy_predicted', inference.get('predicted')), inference.get('energy_target', inference.get('target'))
        all_e = pred + target
        min_e, max_e = min(all_e), max(all_e)
        energy_list = [min_e, max_e]
        target_energy_list = [min_e, max_e]
        for energy, target_energy, graph_index in zip(pred, target, inference['test_frames']):
            phi = phi_psi[graph_index, 0]
            psi = phi_psi[graph_index, 1]
            phi_list.append(phi)
            psi_list.append(psi)
            energy_list.append(energy)
            target_energy_list.append(target_energy)
        error_energy_list = list(np.abs(np.array(energy_list) - np.array(target_energy_list)))
        error_energy_list[:2] = [0, max_e - min_e]
        
        fig, axs = plt.subplots(3, figsize=(8, 12))
        axs[0].scatter(phi_list, psi_list, s=1, c=energy_list, cmap='turbo')
        axs[1].scatter(phi_list, psi_list, s=1, c=error_energy_list, cmap='hot_r')
        axs[2].scatter(phi_list, psi_list, s=1, c=target_energy_list, cmap='turbo')
        dirname = '/'.join(result_file.split("/")[:-1])
        fig.savefig(f'{dirname}/plot.png', dpi=fig.dpi)
        del fig

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FES computation of Alanine Dipeptide using Geometric Deep Learning')
    parser.add_argument('--dir', required=False, help='Directory containing the results to be plotted')

    args = parser.parse_args()
    plot(args.dir, data_dir=DATA_DIR, labels_file=LABELS_FILE)