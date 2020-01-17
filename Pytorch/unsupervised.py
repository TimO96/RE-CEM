from setup_mnist import *
from utils import AE

from torch.optim import SGD, Adam, Adadelta, Adagrad
from torch.nn import MSELoss
from torch import save, cuda, manual_seed
from torch.backends import cudnn
import argparse


class Dataset:
    def __init__(self, data, model, device='cuda:0', random=None):
        """Initialize a dataset."""
        self.data  = data
        self.name  = data.__class__.__name__ + "_" + model.__class__.__name__
        self.model = model.to(device)
        self.device = device

        # Seeding
        if random is not None:
            manual_seed(random)
            if cuda.is_available():
                cudnn.deterministic = True
                cudnn.benchmark = False

    def train(self, epochs=3, optim=Adam, criterion=MSELoss(),
              stats=1000, batch=500, optim_params={'lr':0.01}):
        """Train model with data."""
        optimizer = optim(self.model.parameters(), **optim_params)

        for e in range(epochs):
            X = self.data.train_data

            batch_start, batch_end = 0, batch

            while batch_start < len(X):
                # Prepare new batch.
                img_batch = X[batch_start:batch_end]
                optimizer.zero_grad()

                # Forward + backward + optimize.
                reconstuction = self.model(img_batch)
                loss = criterion(img_batch, reconstuction)
                loss.backward()
                optimizer.step()

                # Print stats.
                if stats: # and batch_start % stats == 0:
                    print(f"{e+1} [{batch_start}|{batch_end}]", end=' ')
                    print(f"loss: {round(loss.item(),3)}")

                # Next batch.
                batch_start = batch_end
                batch_end += batch

    def training(self, batch=64, criterion=MSELoss()):

        X = self.data.train_data
        batch_start, batch_end = 0, batch

        losses = []
        while batch_start < len(X):
            # Prepare new batch.
            img_batch = X[batch_start:batch_end]

            reconstuction = self.model(img_batch)
            losses.append(criterion(img_batch, reconstuction).item())

            # Next batch.
            batch_start = batch_end
            batch_end += batch

        return np.mean(losses)

    def testing(self, batch=64, criterion=MSELoss()):
        X = self.data.test_data
        batch_start, batch_end = 0, batch

        losses = []
        while batch_start < len(X):
            # Prepare new batch.
            img_batch = X[batch_start:batch_end]

            reconstuction = self.model(img_batch)
            losses.append(criterion(img_batch, reconstuction).item())

            # Next batch.
            batch_start = batch_end
            batch_end += batch

        return np.mean(losses)


    def report_performance(self):
        """Performance on all three sets."""
        train, test = self.training(), self.testing()
        print(f"Train: {train} test: {test}")

    def save_model(self, path=None):
        """Store state dict"""
        if path is None:
            path = f'models/{self.name}.pt'
        save(self.model.state_dict(), path)

def search(dset):
    """Sort of grid search."""
    opts = [Adam, SGD, Adadelta, Adagrad]
    for opt in opts:
        for lr in [0.01, 0.001, 0.0001]:
            print(opt, lr)
            train_model(dset, stats=0, optim=opt, optim_params={'lr':lr})

def train_model(dset, **kwargs):
    """Train a specific dataset."""
    device = 'cuda:0' if cuda.is_available() else 'cpu'
    if dset == 'MNIST':
        dataset = Dataset(MNIST(device), AE(), device)
    else:
        raise ModuleNotFoundError(f"Unsupported dataset {d}")

    dataset.train(batch=64, **kwargs)
    dataset.save_model()
    dataset.report_performance()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="MNIST")
    args = vars(parser.parse_args())
    # search(args['dataset'])
    train_model(args['dataset'], stats=0)
