# train.py -- Initialize Dataset class with training functionalities.

# (C) 2020 Changes by UvA FACT AI group [Pytorch conversion]

# Based on:
# Copyright (C) 2018, IBM Corp
#                     Chun-Chen Tu <timtu@umich.edu>
#                     PaiShun Ting <paishun@umich.edu>
#                     Pin-Yu Chen <Pin-Yu.Chen@ibm.com>

from argparse import ArgumentParser
from os import system

from torch import mean, argmax, save, cuda, manual_seed
from torch.optim import SGD, Adam, Adadelta, Adagrad
from torch.nn import CrossEntropyLoss, MSELoss
from torch.backends import cudnn

from models.models import MNISTModel, AE
from data.data import MNIST


class Dataset:
    def __init__(self, data, model, unsupervised, device='cuda:0',
                 criterion=None, seed=None):
        """Initialize a dataset."""
        self.data = data
        self.name = data.type + "_" + model.__class__.__name__
        self.model = model.to(device)
        self.device = device
        self.supervised = not unsupervised
        self.criterion = criterion

        if criterion is None:
            if self.supervised:
                self.criterion = CrossEntropyLoss()
            else:
                self.criterion = MSELoss()

        # Set Pytorch seeds for reproducibility.
        if seed is not None:
            manual_seed(seed)
            if cuda.is_available():
                cudnn.deterministic = True
                cudnn.benchmark = False

    def train(self, epochs=150, optim=Adam, stats=0, batch=128,
              optim_params={'lr': 0.001}):
        """Train model with data."""
        optimizer = optim(self.model.parameters(), **optim_params)

        for epoch in range(epochs):
            X = self.data.train_data
            batch_start, batch_end = 0, batch

            if self.supervised:
                r = argmax(self.data.train_labels, -1)

            while batch_start < len(X):

                # Prepare new batch.
                model_input = X[batch_start:batch_end]
                if self.supervised:
                    labels = r[batch_start:batch_end]
                optimizer.zero_grad()

                # Forward + backward + optimize.
                preds = self.predict(model_input)
                if self.supervised:
                    lss = self.criterion(preds, labels)
                else:
                    lss = self.criterion(model_input, preds)
                lss.backward()
                optimizer.step()

                # Print stats.
                if stats:  # and batch_start % stats == 0:
                    if self.supervised:
                        acc = Dataset.acc(argmax(preds, -1), labels)
                    else:
                        acc = 'NA'
                    print(f"{epoch+1} [{batch_start}|{batch_end}]", end=' ')
                    print(f"loss: {round(lss.item(),3)} acc: {acc}")

                # Next batch.
                batch_start = batch_end
                batch_end += batch

    def p_round(num):
        """Round * 100, 2 dec."""
        return round(num * 100, 2)

    def acc(preds, labels, round=True):
        """Calculate accuracy."""
        acc = mean((preds == labels).float()).item()
        return Dataset.p_round(acc) if round else acc

    def predict(self, batch):
        """Predict output for model for batch."""
        return self.model(batch)

    def batch_data(self, data, labels=None, batch=128):
        """Iterate through the batches and keep track of loss and accuracy."""
        batch_start, batch_end = 0, batch

        lsses, accs = [], []
        while batch_start < len(data):

            # Prepare new batch.
            X = data[batch_start:batch_end]

            preds = self.predict(X)
            if self.supervised:
                r = argmax(labels[batch_start:batch_end], -1)
                lsses.append(self.criterion(preds, r).item())
                accs.append(Dataset.acc(argmax(preds, -1), r, False))
            else:
                lsses.append(self.criterion(X, preds).item())
                accs.append(-0.01)

            # Next batch.
            batch_start = batch_end
            batch_end += batch

        n = len(accs)
        return round(sum(lsses) / n, 3), Dataset.p_round(sum(accs) / n)

    def testing(self):
        """Test model accuracy."""
        return self.batch_data(self.data.test_data, self.data.test_labels)

    def training(self):
        """Get training accuracy."""
        return self.batch_data(self.data.train_data, self.data.train_labels)

    def report_performance(self):
        """Performance on all sets."""
        tlss, tacc = self.training()
        vlss, vacc = self.testing()
        print(f"Train: loss {tlss} acc {tacc}")
        print(f"Test:  loss {vlss} acc {vacc}")

    def save_model(self, path=None, save_dir='models'):
        """Store state dict."""
        if path is None:
            system(f"mkdir -p {save_dir}")
            path = f'{save_dir}/{self.name}.pt'
        save(self.model.state_dict(), path)
        print(f'Stored to {path}')


def search(dset, unsupervised):
    """Sort of grid search."""
    opts = [Adam, SGD, Adagrad, Adadelta]
    for opt in opts:
        for lr in [0.01, 0.001]:
            for b in [32, 64, 100, 128, 256]:
                print(opt, lr, b)
                for seed in [10, 82, 43, 398, 112]:
                    train_model(dset, unsupervised, seed, batch=b, epochs=150,
                                optim=opt, optim_params={'lr': lr})


def train_model(dset, unsupervised, seed=None, **kwargs):
    """Train a specific dataset."""
    dvc = 'cuda:0' if cuda.is_available() else 'cpu'

    if dset in ('MNIST', 'FMNIST'):
        model = AE() if unsupervised else MNISTModel()
        dataset = Dataset(MNIST(dvc, dset), model, unsupervised, dvc,
                          seed=seed)
    else:
        raise ModuleNotFoundError(f"Unsupported dataset {d}")

    dataset.train(**kwargs)
    dataset.save_model()
    dataset.report_performance()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="MNIST")
    parser.add_argument("-s", "--search", type=int, default=0)
    parser.add_argument("-u", "--unsupervised", type=bool, default=False)
    args = vars(parser.parse_args())

    d, u = args['dataset'], args['unsupervised']
    if args['search']:
        search(d, u)
    else:
        train_model(d, u, batch=128, stats=1000, epochs=1)
