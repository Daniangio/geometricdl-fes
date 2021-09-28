# echo temp1234 | horovodrun -np 1 sudo -S python3 train.py

import argparse
import json
import os
from filelock import FileLock
import tempfile
import random

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler

from torch_geometric.loader import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import horovod.torch as hvd

from models.linear import LinearNet
from dataset import Dataset, create_transform

# Training settings
parser = argparse.ArgumentParser(description='FES computation of Alanine Dipeptide using Geometric Deep Learning')
parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                    help='input batch size for training (default: 4)')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 32)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1304, metavar='S',
                    help='random seed (default: 1304)')
parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')
parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                    help='apply gradient predivide factor in optimizer (default: 1.0)')
parser.add_argument('--data-dir', default='/scratch/carrad',
                    help='location of the training dataset in the local filesystem (will be downloaded if needed)')


def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()

def load_indexes(data_dir, n_samples, test_on):
    print('loading indexes')
    with open(f'{data_dir}/left.json', 'r') as l:
        left = json.load(l)

    with open(f"{data_dir}/right.json", "r") as r:
        right = json.load(r)

    if test_on == 'all_but_left':
        train_ind = left[::2] # take 50% of left area as training
        remaining_ind = [i for i in range(n_samples) if i not in train_ind]
    elif test_on == 'all_but_right':
        train_ind = right[::2] # take 50% of right area as training
        remaining_ind = [i for i in range(n_samples) if i not in train_ind]
    elif test_on == 'left':
        remaining_ind = left
        train_ind = [i for i in range(n_samples) if i not in remaining_ind][::5] # take 20% of non-left area as training
    elif test_on == 'right':
        remaining_ind = right
        train_ind = [i for i in range(n_samples) if i not in remaining_ind][::5] # take 20% of non-right area as training

    random.shuffle(train_ind)
    random.shuffle(remaining_ind)

    split = int(0.1 * len(remaining_ind))
    validation_ind = remaining_ind[:split]
    test_ind = remaining_ind[split:]
    print('indexes loaded')
    return train_ind, validation_ind, test_ind


# def test():
#     model.eval()
#     test_loss = 0.
#     test_accuracy = 0.
#     for data, target in test_loader:
#         if args.cuda:
#             data, target = data.cuda(), target.cuda()
#         output = model(data)
#         # sum up batch loss
#         test_loss += F.nll_loss(output, target, size_average=False).item()
#         # get the index of the max log-probability
#         pred = output.data.max(1, keepdim=True)[1]
#         test_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum()

#     # Horovod: use test_sampler to determine the number of examples in
#     # this worker's partition.
#     test_loss /= len(test_sampler)
#     test_accuracy /= len(test_sampler)

#     # Horovod: average metric values across workers.
#     test_loss = metric_average(test_loss, 'avg_loss')
#     test_accuracy = metric_average(test_accuracy, 'avg_accuracy')

#     # Horovod: print output only on first rank.
#     if hvd.rank() == 0:
#         print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
#             test_loss, 100. * test_accuracy))


if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    hvd.init()

    kwargs = {'num_workers': 2}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    # get data
    data_dir = args.data_dir
    n_samples = 5000
    with FileLock(os.path.expanduser("~/.horovod_lock")):
        train_indexes, validation_indexes, test_indexes = load_indexes(data_dir, n_samples, 'right')

        transform = create_transform(n_samples)
        train_dataset = Dataset(data_dir=data_dir, indexes=train_indexes, transform=transform)

    # set training data loader
    train_sampler = DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)


    epochs = args.epochs
    with tempfile.TemporaryDirectory() as run_output_dir:
        ckpt_path = os.path.join(run_output_dir, "checkpoint")
        os.makedirs(ckpt_path, exist_ok=True)

        logs_path = os.path.join(run_output_dir, "logger")
        os.makedirs(logs_path, exist_ok=True)
        logger = TensorBoardLogger(logs_path)

        train_percent = 1.0
        val_percent = 1.0

        model = LinearNet(next(iter(train_loader)), args.lr, args.momentum)
        setattr(model, 'train_dataloader', lambda: train_loader)
        # setattr(model, 'val_dataloader', lambda: test_loader)

        from pytorch_lightning.callbacks import Callback

        class MyDummyCallback(Callback):
            def __init__(self):
                self.epcoh_end_counter = 0
                self.train_epcoh_end_counter = 0

            def on_init_start(self, trainer):
                print('Starting to init trainer!')

            def on_init_end(self, trainer):
                print('Trainer is initialized.')

            def on_epoch_end(self, trainer, model):
                print('A epoch ended.')
                self.epcoh_end_counter += 1

            def on_train_epoch_end(self, trainer, model, unused=None):
                print('A train epoch ended.')
                self.train_epcoh_end_counter += 1

            def on_train_end(self, trainer, model):
                print('Training ends')
                assert self.epcoh_end_counter == 2 * epochs
                assert self.train_epcoh_end_counter == epochs

        callbacks = [MyDummyCallback(), ModelCheckpoint(dirpath=ckpt_path)]

        trainer = Trainer(accelerator='horovod',
                          gpus=(1 if torch.cuda.is_available() else 0),
                          callbacks=callbacks,
                          max_epochs=epochs,
                          limit_train_batches=train_percent,
                          limit_val_batches=val_percent,
                          logger=logger,
                          num_sanity_val_steps=0)

        trainer.fit(model)

        # test()