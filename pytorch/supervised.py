## (C) 2020 UvA FACT AI group

from setup_mnist import *
from torch.optim import SGD, Adam, Adadelta, Adagrad
from torch.nn import CrossEntropyLoss
from torch import mean as torch_mean
from torch import argmax as torch_argmax
from torch import save, cuda, manual_seed
from torch.backends import cudnn
import argparse


class Dataset:
    def __init__(self, data, model, device='cuda:0', seed=None):
        """Initialize a dataset."""

        self.data  = data
        self.name  = data.__class__.__name__ + "_" + model.__class__.__name__
        self.model = model.to(device)
        self.device = device

        # Set Pytorch seeds for reproducibility.
        if seed is not None:
            manual_seed(seed)
            if cuda.is_available():
                cudnn.deterministic = True
                cudnn.benchmark = False

    def train(self, epochs=150, optim=Adam, criterion=CrossEntropyLoss(),
              stats=0, batch=128, optim_params={'lr':0.001}):
        """Train model with data."""

        optimizer = optim(self.model.parameters(), **optim_params)

        for e in range(epochs):
            X = self.data.train_data
            r = torch_argmax(self.data.train_labels, -1)
            batch_start, batch_end = 0, batch

            while batch_start < len(X):
                
                # Prepare new batch.
                input = X[batch_start:batch_end]
                labels = r[batch_start:batch_end]
                optimizer.zero_grad()

                # Forward + backward + optimize.
                preds = self.predict(input)
                lss = criterion(preds, labels)
                lss.backward()
                optimizer.step()

                # Print stats.
                if stats: # and batch_start % stats == 0:
                    acc = Dataset.acc(torch_argmax(preds, -1), labels)
                    print(f"{e+1} [{batch_start}|{batch_end}]", end=' ')
                    print(f"loss: {round(lss.item(),3)} acc: {acc}")

                # Next batch.
                batch_start = batch_end
                batch_end += batch

    def acc(preds, labels):
        """Calculate accuracy."""
        return round(torch_mean((preds == labels).float()).item() * 100, 3)

    def predict(self, batch):
        """Predict output for model for batch."""
        return self.model(batch)

    def test(self, X, r, label=None, p=False):
        """Test the accuracy of X on model (labelled), optionally print."""
        preds = torch_argmax(self.predict(X), -1)
        acc = Dataset.acc(preds, torch_argmax(r, -1))
        if p: (f'\n{label} acc: {acc}')
        return acc

    def testing(self, p=False):
        """Test model accuracy."""
        return self.test(self.data.test_data, self.data.test_labels, 'Test', p)

    def training(self, p=False):
        """Get training accuracy."""
        X,r = self.data.train_data, self.data.train_labels
        s2 = round(len(r) / 2)
        s1 = round(s2 / 2)
        s3 = s2 + s1

        # Batch for memory
        t1 = self.test(X[:s1], r[:s1], 'Training', p)
        t2 = self.test(X[s1:s2], r[s1:s2], 'Training', p)
        t3 = self.test(X[s2:s3], r[s2:s3], 'Training', p)
        t4 = self.test(X[s3:], r[s3:], 'Training', p)

        return round((t1 + t2 + t3 + t3) / 4, 3)

    def report_performance(self):
        """Performance on all sets."""
        print(f"Train: {self.training()} test: {self.testing()}")

    def save_model(self, path=None):
        """Store state dict"""
        if path is None:
            path = f'models/{self.name}.pt'
        save(self.model.state_dict(), path)

def search(dset):
    """Sort of grid search."""
    opts = [Adam, Adagrad]
    for opt in opts:
        for lr in [0.01, 0.001]:
            for b in [32, 64, 100, 128]:
                print(opt, lr, b)
                for seed in [10, 82, 43, 398, 112]:
                    train_model(dset, seed, batch=b, epochs=150, optim=opt,
                                optim_params={'lr':lr})

def train_model(dset, seed=None, **kwargs):
    """Train a specific dataset."""
    device = 'cuda:0' if cuda.is_available() else 'cpu'

    if dset == 'MNIST':
        dataset = Dataset(MNIST(device), MNISTModel(), device, seed=seed)
    else:
        raise ModuleNotFoundError(f"Unsupported dataset {d}")

    dataset.train(**kwargs)
    dataset.save_model()
    dataset.report_performance()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="MNIST")
    parser.add_argument("-s", "--search", type=int, default=0)
    args = vars(parser.parse_args())

    if args['search']:
        search(args['dataset'])
    else:
        train_model(args['dataset'], batch=128)
