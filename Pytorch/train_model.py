from setup_mnist import *
from torch.optim import SGD,Adam
from torch.nn import CrossEntropyLoss
from torch import mean as torch_mean
from torch import argmax as torch_argmax
from torch import save, cuda, manual_seed
from torch.backends import cudnn
import argparse

class Dataset:
    def __init__(self, data, model, device='cuda:0', lr=0.01, random=123):
        """Initialize a dataset."""
        self.data  = data
        self.name  = data.__class__.__name__
        self.model = model.to(device)
        self.lr = lr
        self.device = device

        manual_seed(random)
        if cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def train(self, epochs=3, optimizer=None, criterion=CrossEntropyLoss(),
              stats=1000, batch=500):
        """Train model with data."""
        if optimizer is None:
            optimizer = Adam(self.model.parameters(), lr=self.lr)

        for e in range(epochs):
            X = self.data.train_data
            r = torch_argmax(self.data.train_labels, -1)
            batch_start, batch_end = 0, batch

            while batch_start <= len(X):
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
                if stats and batch_start % stats == 0:
                    acc = Dataset.acc(torch_argmax(preds, -1), labels)
                    print(f"{e+1} [{batch_start}|{batch_end}] loss: {lss} acc: {acc}")

                # Next batch.
                batch_start = batch_end
                batch_end += batch

    def performance_test(self, X, r, label):
        """ """
        preds = torch_argmax(self.predict(X), -1)
        acc = Dataset.acc(preds, torch_argmax(r, -1))
        print(f'\n{label} acc: {acc}\n')
        return acc

    def test(self):
        """Test model accuracy."""
        return self.performance_test(self.data.test_data, self.data.test_labels, 'Test')

    def validate(self):
        """Validate model accuracy."""
        return self.performance_test(self.data.validation_data, self.data.validation_labels, 'Validation')

    def acc(preds, labels):
        """Calculate accuracy."""
        return round(torch_mean((preds == labels).float()).item(), 3)

    def predict(self, batch):
        """Predict output for model for batch."""
        return self.model(batch)

    def save_model(self, path=None):
        """Store state dict"""
        if path is None:
            path = f'models/{self.name}.pt'
        save(self.model.state_dict(), path)

def search(args):
    train_model(args)


def train_model(args):
    """Train a specific dataset."""
    device = 'cuda' if cuda.is_available() else 'cpu'
    d = args['dataset']
    if d == 'MNIST':
        dataset = Dataset(MNIST(device), MNISTModel(), device, lr=0.001)
    else:
        raise ModuleNotFoundError(f"Unsupported dataset {d}")

    dataset.train(batch=100)
    dataset.save_model()
    dataset.test()
    dataset.validate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="MNIST")
    # parser.add_argument("-m", "--maxiter", type=int, default=1000)
    # parser.add_argument("-b", "--binary_steps", type=int, default=9)
    # parser.add_argument("-c", "--init_const", type=float, default=10.0)
    # parser.add_argument("--mode", choices=["PN", "PP"], default="PN")
    # parser.add_argument("--kappa", type=float, default=0)
    # parser.add_argument("--beta", type=float, default=1e-1)
    # parser.add_argument("--gamma", type=float, default=0)

    args = vars(parser.parse_args())
    search(args)
