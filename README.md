# FACT-AI CEM-I
> [RE] Explanations based on the Missing: Towards Contrastive Explanations with Pertinent Negatives

## Authors
- Tim Ottens (11147598)
- Stefan Klut (11331720)
- Thomas van Osch (11248815)
- Mattijs Blankesteijn (11318422)

## Instructions
See `example.ipynb` for example usage of the `cem` class. Alternatively, `main.py` can be called with the following arguments:

```
usage: main.py [-h] [--dataset DATASET] [--seed SEED] [--id ID] [--mode MODE]
               [--max_iter MAX_ITER] [--gamma GAMMA] [--kappa KAPPA]
               [--quant_eval QUANT_EVAL] [--n_samples N_SAMPLES] [-s SEARCH]
               [-u UNSUPERVISED]

Contrastive Explanations Method (CEM)

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Used dataset - either MNIST or FMNIST
  --seed SEED           Seed for reproducablitity
  --id ID               Id of the used image
  --mode MODE           Type of pertubation
  --max_iter MAX_ITER   Type of pertubation
  --gamma GAMMA         Hyperparameter for the effect of the autoencoder
  --kappa KAPPA         Hyperparameter for the desired confidence
  --quant_eval QUANT_EVAL
                        Run the quantative evaluation
  --n_samples N_SAMPLES
                        Number of samples for quantative evaluation
  -s SEARCH, --search SEARCH
                        Search for best training hyperparameters
  -u UNSUPERVISED, --unsupervised UNSUPERVISED
                        True trains an autoencoder firstly, False trains an NN
                        model firstly.
```

## Acknowledgments
This project was supervised by [Maurice (Morris) Frank](https://morris-frank.dev/).

