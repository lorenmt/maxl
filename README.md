# Meta AuXiliary Learning [MAXL]

*In the process of cleaning ... not final version yet. Please check back soon.*

This repository contains the source code to support the paper: [Self-Supervised Generalisation with Meta Auxiliary Learning](https://arxiv.org/abs/1901.08933), introduced by [Shikun Liu](http://shikun.io/), [Andrew J. Davison](http://www.doc.ic.ac.uk/~ajd/) and [Edward Johns](https://www.robot-learning.uk/).

## Requirements
MAXL was written in `python 3.7` and `pytorch 1.0`. We recommend running the code through the same version while we believe the code should also work (or can be easily revised) within other versions.


## Models & Datasets
This repository includes three models `model_vgg_single.py`, `model_vgg_human.py` and `model_vgg_maxl.py` representing baselines `Single`, `Human` and our proposed algorithm `MAXL` with backbone architecture `VGG-16`. Those three models are trained with `4-level CIFAR-100 dataset` which should easily reproduce part of the results in Figure 3.

In `create_dataset.py`, we define an extended version of CIFAR-100 with 4-level hierarchy built on the original CIFAR100 class in `torchvision.dataset`. To fetch one batch of data as defined below, we have `train_data` which represents the input images and `train_label` which represents the 4-level hiearchy labels with the order in 3, 10, 20, 100 classes.

```
train_data, train_label = cifar100_train_dataset.next()
```

## Training MAXL
The source code provided gives an example of training primary task of 20 classes and auxiliary task of 100 classes. You may revise the code easily to evaluate other hierarchies or you may also evaluate other datasets.

Training MAXL typically requires 30 hours from scratch (computing the Hessian matrix is slow), and training the baselines methods `Single` and `Human` requires 4 hours from scratch.


## Citation
If you found this code/work to be useful in your own research, please considering citing the following:

```
@article{liu2019maxl,
  title={Self-Supervised Generalisation with Meta Auxiliary Learning},
  author={Liu, Shikun and Davison, Andrew J and Johns, Edward},
  journal={arXiv preprint arXiv:1901.08933},
  year={2019}
}
```

## Contact
If you have any questions, please contact `sk.lorenmt@gmail.com`.
