# MAXL - Meta Auxiliary Learning
This repository contains the source code to support the paper: [Self-Supervised Generalisation with Meta Auxiliary Learning](https://arxiv.org/abs/1901.08933), introduced by [Shikun Liu](http://shikun.io/), [Andrew J. Davison](http://www.doc.ic.ac.uk/~ajd/) and [Edward Johns](https://www.robot-learning.uk/).

See project page [here](https://shikun.io/projects/meta-auxiliary-learning).

## Update
**Nov 2021**: We have implemented the first order approximation of MAXL framework, which would speed up 4 - 6 times training time compared to the original implementation. The first order approximation is based on the finite difference method, inspired by [DARTS](https://arxiv.org/abs/1806.09055). No more tedious forward functions for the inner loop optimisation now. Enjoy. :)

## Requirements
MAXL was written in `python 3.7` and `pytorch 1.0`. We recommend running the code through the same version while we believe the code should also work (or can be easily revised) within other versions.


## Models & Datasets
This repository includes three models `model_vgg_single.py`, `model_vgg_human.py` and `model_vgg_maxl.py` representing baselines `Single`, `Human` and our proposed algorithm `MAXL` with backbone architecture `VGG-16`. These three models are trained with `4-level CIFAR-100 dataset` which should easily reproduce part of the results in Figure 3.

In `create_dataset.py`, we define an extended version of CIFAR-100 with 4-level hierarchy built on the original CIFAR100 class in `torchvision.datasets` (see the full table for semantic classes in Appendix A). To fetch one batch of input data with kth hierarchical labels as defined below, we have `train_data` which represents the input images and `train_label` which represents the 4-level hierarchical labels: `train_label[:, k], k = 0, 1, 2, 3` fetches 3, 10, 20 and 100-classes respectively.

```
train_data, train_label[:, k] = cifar100_train_dataset.next()
```

## Training MAXL
The source code provided gives an example of training primary task of 20 classes `train_label[:, 2]` and auxiliary task of 100 classes `train_label[:, 3]` with hierarchical structure `\psi[i]=5`. To run the code, please create a folder `dataset` to download CIFAR-100 dataset in this directory or you may redefine the dataset root path as your wish. It is straightforward to revise the code evaluating other hierarchies and play with other datasets found in `torchvision.datasets`.

Note that: make sure `len(psi)` be the number of primary classes, and `sum(psi)` be the number of total auxiliary classes, e.g. `psi = [2,3,4]` representing total 3 primary classes and total 9 auxiliary classes by splitting each corresponding primary class into 2, 3, and 4 different auxiliary classes.

Training MAXL from scratch typically requires 30 hours in GTX 1080, and training the baselines methods `Single` and `Human` requires 2-4 hours from scratch.

## Citation
If you found this code/work to be useful in your own research, please considering citing the following:

```
@inproceedings{liu2019maxl,
  title={Self-supervised generalisation with meta auxiliary learning},
  author={Liu, Shikun and Davison, Andrew and Johns, Edward},
  booktitle={Advances in Neural Information Processing Systems},
  pages={1677--1687},
  year={2019}
}
```

## Contact
If you have any questions, please contact `sk.lorenmt@gmail.com`.
