# main.py -- main file with arguments
# (C) 2020 UvA FACT AI

import random
import argparse
import cem
from cem.train import search, train_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Contrastive Explanations\
                                                  Method (CEM)')
    parser.add_argument('--dataset', type=str, default="MNIST", help='Used\
                        dataset - either MNIST or FMNIST')
    parser.add_argument('--seed', type=int, default=121, help='Seed for\
                        reproducablitity')
    parser.add_argument('--id', type=int, default=0, help='Id of the used\
                        image')
    parser.add_argument('--mode', type=str, default="PN", help='Type of\
                        pertubation')
    parser.add_argument('--max_iter', type=str, default=1000, help='Type of\
                        pertubation')
    parser.add_argument('--gamma', type=int, default=100, help='Hyperparameter\
                        for the effect of the autoencoder')
    parser.add_argument('--kappa', type=int, default=10, help='Hyperparameter\
                        for the desired confidence')
    parser.add_argument('--quant_eval', type=bool, default=False, help='Run\
                        the quantative evaluation')
    parser.add_argument('--n_samples', type=int, default=400, help='Number of\
                        samples for quantative evaluation')
    parser.add_argument("-s", "--search", type=int, default=0, help='Search\
                        for best training hyperparameters')
    parser.add_argument("-u", "--unsupervised", type=bool, default=None,
                        help='True trains an autoencoder firstly, False trains\
                              an NN model firstly.')
    args = parser.parse_args()

    # Train optionally
    if args.unsupervised is not None:
        if args.search:
            search(args.dataset, args.unsupervised)
        else:
            train_model(args.dataset, args.unsupervised, stats=1000)

    # Perform explanation
    m = cem.Main(seed=args.seed, type=args.dataset)
    if args.quant_eval:
        print(m.quant_eval(ids=random.sample(range(10000), args.n_samples)))
    else:
        m.explain(args.id, mode=args.mode, max_iter=args.max_iter,
                  gamma=args.gamma, kappa=args.kappa)
