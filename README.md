# FACT-AI CEM-I
> [RE] Explanations based on the Missing: Towards Contrastive Explanations with Pertinent Negatives

Reproducing
> Amit Dhurandhar, Pin-Yu Chen, Ronny Luss, Chun-Chen Tu, Paishun Ting,
Karthikeyan Shanmugam, and Payel Das. 2018. Explanations based on
the Missing: Towards Contrastive Explanations with Pertinent Negatives.
[arXiv:cs.AI/1802.07623](https://arxiv.org/abs/1802.07623)

## Authors
- Tim Ottens (11147598) [t_im1996@live.nl](mailto:t_im1996@live.nl)
- Stefan Klut (11331720) [stefanklut12@gmail.com](mailto:stefanklut12@gmail.com)
- Thomas van Osch (11248815) [t.vanosch@hotmail.com](mailto:t.vanosch@hotmail.com)
- Mattijs Blankesteijn (11318422) [re_cem@mattijsblankesteijn.nl](mailto:re_cem@mattijsblankesteijn.nl)

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

