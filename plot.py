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
LABELS_FILE_MD = 'phi-psi-free-energy-approx.txt'

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


def plot_from_md(data_dir: str, labels_file: str):
    with open(os.path.join(data_dir, labels_file), "r") as t:
        labels = torch.Tensor([[float(phi), float(psi), float(fe)] for _, phi, psi, fe in [x.split() for x in t.readlines()]])
        labels = labels[labels[:, -1] > 0]
        labels[:, :-1] = labels[:, :-1] / np.pi * 180
        # labels[:, -1] = labels[:, -1] / torch.sum(labels[:, -1])
        # K = 1
        # labels[:, -1] = -torch.log(labels[:, -1]) * K

    phi_list = []
    psi_list = []
    energy_list = []
    for i in range(len(labels)):
        phi = labels[i, 0]
        psi = labels[i, 1]
        phi_list.append(phi)
        psi_list.append(psi)
        energy_list.append(labels[i, 2])
    
    fig, axs = plt.subplots(1, figsize=(8, 12))
    plt.scatter(phi_list, psi_list, s=1, c=energy_list, cmap='turbo', label=energy_list)
    plt.colorbar()
    fig.savefig(f'approx-plot.png', dpi=fig.dpi)
    del fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FES computation of Alanine Dipeptide using Geometric Deep Learning')
    parser.add_argument('--dir', required=False, help='Directory containing the results to be plotted')

    args = parser.parse_args()
    if args.dir is None:
        plot_from_md( data_dir=DATA_DIR, labels_file=LABELS_FILE_MD)
    else:
        plot(args.dir, data_dir=DATA_DIR, labels_file=LABELS_FILE)