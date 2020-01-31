# main.py -- main file with arguments
# (C) 2020 UvA FACT AI

import cem
import random

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Contrastive Explanations Method (CEM)')

    parser.add_argument('--dataset', type=str, default="MNIST", help='Used dataset - either MNIST or FMNIST')
    parser.add_argument('--seed', type=int, default=121, help='Seed for reproducablitity')
    parser.add_argument('--id', type=int, default=0, help='Id of the used test image')
    parser.add_argument('--mode', type=str, default="PN", help='Type of pertubation')
    parser.add_argument('--max_iter', type=str, default=1000, help='Type of pertubation')
    parser.add_argument('--gamma', type=int, default=100, help='Hyperparameter for the effect of the autoencoder')
    parser.add_argument('--kappa', type=int, default=10, help='Hyperparameter for the desired confidence')
    parser.add_argument('--quant_eval', type=bool, default=False, help='Run the quantative evaluation')
    parser.add_argument('--n_samples', type=int, default=400, help='Number of samples for quantative evaluation')

    args = parser.parse_args()

    m = cem.Main(seed=args.seed, type=args.dataset)

    if args.quant_eval:
        print(m.quant_eval(ids=random.sample(range(10000), args.n_samples)))
    else:
        m.explain(args.id, mode=args.mode, max_iter=args.max_iter, gamma=args.gamma, kappa=args.kappa)
