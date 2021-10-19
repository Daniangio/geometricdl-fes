# horovodrun -np 2 python3 train.py --epoch 30 --weights results/right/linear-old-0929-1139-mae\:0.07/parameters.pt

import argparse
from datetime import datetime
import json
import os
from typing import Union
from filelock import FileLock
import shutil
from pathlib import Path
import random
from pytorch_lightning.core.lightning import LightningModule

import torch
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import horovod.torch as hvd
from models.geometric import GeometricNet
from models.geometricphipsi import GeometricPhiPsiNet

from models.linear import LinearNet
from models.graphdihedrals import BISGraphConvPoolNet, GraphConvPoolNet
from dataset import Dataset, create_transform

from torch_geometric.loader import DataLoader
from ray.tune.integration.pytorch_lightning import TuneReportCallback

from plot import plot

# Training settings
parser = argparse.ArgumentParser(description='FES computation of Alanine Dipeptide using Geometric Deep Learning')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 32)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=Union[int, None], default=None, metavar='S',
                    help='use manual seed (default: Do not use)')
parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')
parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                    help='apply gradient predivide factor in optimizer (default: 1.0)')
parser.add_argument('--data-dir', default='data/ala_dipep',
                    help='location of the training dataset in the local filesystem')
parser.add_argument('--labels-file', default='data/ala_dipep/phi-psi-free-energy.txt',
                    help='file containing the energy values associated with each frame')
parser.add_argument('--weights', default=None,
                    help='file containing the model weights to be loaded before training')
parser.add_argument('--checkpoint', default=None,
                    help='path to the checkpoint file of a training that did not finish')
parser.add_argument('--test-on', nargs='+', help='<Required> Portion of the FES where to test inference, train on all the remaining FES.' +
                    '\nPossible values: left, right, all_but_left, all_but_right', required=True)
parser.add_argument('--model', default='graph', help='Name of the model to use')


def metric_average(val, name):
    avg_tensor = hvd.allreduce(val, name=name)
    return avg_tensor.item()

def gather(val, name):
    tensor = torch.Tensor(val)
    gathered_tensor = hvd.allgather(tensor, name=name)
    return gathered_tensor

def load_indexes(data_dir, test_on, n_samples=50000):
    print('loading indexes')
    with open(os.path.join(data_dir, 'left.json'), 'r') as l:
        left = json.load(l)
        random.shuffle(left)

    with open(os.path.join(data_dir, 'right.json'), 'r') as r:
        right = json.load(r)
        random.shuffle(right)

    if test_on == 'all_but_left':
        train_ind = left[::3] # take 33% of left area as training
        remaining_ind = [i for i in range(n_samples) if i not in left]
        random.shuffle(remaining_ind)
    elif test_on == 'all_but_right':
        train_ind = right[::2] # take 50% of right area as training
        remaining_ind = [i for i in range(n_samples) if i not in right]
        random.shuffle(remaining_ind)
    elif test_on == 'left':
        remaining_ind = left
        train_ind = [i for i in range(n_samples) if i not in remaining_ind]
        random.shuffle(train_ind)
        train_ind = train_ind[::5] # take 20% of non-left area as training
    elif test_on == 'right':
        remaining_ind = right
        train_ind = [i for i in range(n_samples) if i not in remaining_ind]
        random.shuffle(train_ind)
        train_ind = train_ind[::5] # take 20% of non-right area as training
    elif test_on == 'all':
        all_indexes = [i for i in range(n_samples)]
        random.shuffle(all_indexes)
        train_ind = all_indexes[::5]
        remaining_ind = [i for i in range(n_samples) if i not in train_ind]

    split = int(0.1 * len(remaining_ind))
    validation_ind = remaining_ind[:split]
    test_ind = remaining_ind[split:]
    print('indexes loaded')
    return train_ind, validation_ind, test_ind


def test(model: LightningModule, test_loader, test_sampler):
    model = model.cuda()
    model.eval()
    test_loss = 0.
    energy_predictions, energy_targets, dihedrals_predictions, dihedrals_targets = [], [], [], []
    graph_indexes = []
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            if args.cuda:
                data = data.cuda()
            # run inference and sum up batch loss
            results = model.test_step(data, idx)
            test_loss += results.get('test_loss')
            graph_indexes.extend(results.get('graph_indexes'))
            energy_predictions.extend(results.get('energy_predictions'))
            energy_targets.extend(results.get('energy_targets'))
            if results.get('dihedrals_predictions') is not None:
                dihedrals_predictions.extend(results.get('dihedrals_predictions'))
                dihedrals_targets.extend(results.get('dihedrals_targets'))
            

        # Horovod: use test_sampler to determine the number of examples in
        # this worker's partition.
        test_loss /= len(test_sampler)

        # Horovod: average metric values across workers.
        test_loss = metric_average(test_loss, 'avg_loss')
        energy_predictions = gather(energy_predictions, 'all_energy_predictions')
        energy_targets = gather(energy_targets, 'all_energy_targets')
        dihedrals_predictions = gather(dihedrals_predictions, 'all_dihedrals_predictions')
        dihedrals_targets = gather(dihedrals_targets, 'all_dihedrals_targets')

        # Horovod: print output only on first rank.
        if hvd.rank() == 0:
            print(f'\nTest set: Average loss: {test_loss}\n')

        return {
            'test_acc': test_loss,
            'test_loss': test_loss,
            'dihedrals_predictions': dihedrals_predictions,
            'dihedrals_targets': dihedrals_targets,
            'energy_predictions': energy_predictions,
            'energy_targets': energy_targets,
            'graph_indexes': graph_indexes
        }


def save_results(model, train_indexes, test_indexes, test_results: dict, test_on, args):
    directory = f"results/{test_on}/{model.name}-{datetime.now().strftime('%m%d-%H%M')}-{model.test_criterion_initials}:{test_results.get('test_acc'):.2f}"
    print(directory)
    graph_indexes = [int(idx) for idx in test_results.get('graph_indexes')]
    sorted_energy_predictions = [int(x.item()) for _, x in sorted(zip(graph_indexes, test_results.get('energy_predictions')))]
    sorted_energy_targets = [int(x.item()) for _, x in sorted(zip(graph_indexes, test_results.get('energy_targets')))]
    result = {
            "train_parameters": {
                "batch-size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "seed": args.seed
            },
            "test_accuracy": test_results.get('test_acc'),
            "test_loss": test_results.get('test_loss'),
            "energy_predicted": sorted_energy_predictions,
            "energy_target": sorted_energy_targets,
            "test_frames": sorted(graph_indexes),
            "train_frames": train_indexes,
        }
    
    if test_results.get('dihedrals_predictions') is not None:
        sorted_dihedrals_predictions = [x.cpu().tolist() for _, x in sorted(zip(graph_indexes, test_results.get('dihedrals_predictions')))]
        sorted_dihedrals_targets = [x.cpu().tolist() for _, x in sorted(zip(graph_indexes, test_results.get('dihedrals_targets')))]
        result.update({
            "dihedrals_predicted": sorted_dihedrals_predictions,
            "dihedrals_target": sorted_dihedrals_targets
        })

    os.makedirs(directory)
    with open(f"{directory}/result.json", "w") as f:
        json.dump(result, f)

    torch.save(model.state_dict(), f"{directory}/parameters.pt")
    plot(f"{directory}/result.json")


if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    hvd.init()

    kwargs = {'num_workers': 12}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    if args.seed:
        torch.manual_seed(args.seed)
    
    models = {
        'linear': (LinearNet, True), # Model, use_dihedrals
        'graph': (GraphConvPoolNet, True),
        'geometric': (GeometricNet, False),
        'geometricphipsi': (GeometricPhiPsiNet, False)
    }

    # get data
    data_dir = args.data_dir
    labels_file = args.labels_file

    for test_on in args.test_on:
        with FileLock(os.path.expanduser("~/.horovod_lock")):
            train_indexes, validation_indexes, test_indexes = load_indexes(data_dir, test_on)

            transform = create_transform(data_dir, labels_file, use_dihedrals=models[args.model][1])
            train_dataset = Dataset(data_dir=data_dir, indexes=train_indexes, transform=transform, use_dihedrals=models[args.model][1])
            test_dataset = Dataset(data_dir=data_dir, indexes=test_indexes, transform=transform, use_dihedrals=models[args.model][1])

        # set train data loader
        train_sampler = DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)

        # set test data loader
        test_sampler = DistributedSampler(test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, sampler=test_sampler, **kwargs)

        epochs = args.epochs
        if args.checkpoint:
            run_output_dir = '/'.join(x.strip() for x in  args.checkpoint.split('/')[:-2])
        else:
            run_output_dir = f"tmpdir-{test_on}-{datetime.now().strftime('%m%d-%H%M')}"
            Path(run_output_dir).mkdir(exist_ok=True)
        ckpt_path = os.path.join(run_output_dir, "checkpoint")
        logs_path = os.path.join(run_output_dir, "logger")

        if not args.checkpoint:
            os.makedirs(ckpt_path, exist_ok=True)
            os.makedirs(logs_path, exist_ok=True)
        logger = TensorBoardLogger(logs_path)

        train_percent = 1.0
        val_percent = 1.0

        if args.checkpoint:
            assert not args.weights, 'Params --weights and --checkpoint are mutually exclusive'
            print(f'Loading model from checkpoint: {args.checkpoint}')
            model = models[args.model][0].load_from_checkpoint(checkpoint_path=args.checkpoint, sample=next(iter(train_loader)), lr=args.lr)
        else:
            model = models[args.model][0](next(iter(train_loader)), args.lr)
            if args.weights:
                try:
                    model.load_state_dict(torch.load(args.weights), strict=False)
                    print(f'Model weights {args.weights} loaded')
                except Exception as e:
                    print(f'Model weights could not be loaded: {str(e)}')
        print(model)
        
        setattr(model, 'train_dataloader', lambda: train_loader)
        # setattr(model, 'val_dataloader', lambda: test_loader)

        from pytorch_lightning.callbacks import Callback

        class MyDummyCallback(Callback):
            def __init__(self):
                pass

            def on_init_start(self, trainer):
                print('Starting to init trainer!')

            def on_init_end(self, trainer):
                print('Trainer is initialized.')

            def on_epoch_end(self, trainer, model):
                pass # print('A epoch ended.')

            def on_train_epoch_end(self, trainer, model, unused=None):
                pass # print('A train epoch ended.')

            def on_train_end(self, trainer, model):
                pass # print('Training ends')

        callbacks = [MyDummyCallback(), ModelCheckpoint(dirpath=ckpt_path)]

        trainer = Trainer(accelerator=None,
                        gpus=(1 if (args.cuda and torch.cuda.is_available()) else 0),
                        callbacks=callbacks,
                        max_epochs=epochs,
                        limit_train_batches=train_percent,
                        limit_val_batches=val_percent,
                        logger=logger,
                        num_sanity_val_steps=0)

        trainer.fit(model)

        test_results = test(model, test_loader, test_sampler)
        if hvd.rank() == 0:
            save_results(model, train_indexes, test_indexes, test_results, test_on, args)
        #shutil.rmtree(run_output_dir, ignore_errors=True)