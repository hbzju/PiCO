# PiCO+: Contrastive Label Disambiguation for Robust Partial Label Learning

This is a [PyTorch](http://pytorch.org) implementation of [PiCO+](https://arxiv.org/pdf/2201.08984.pdf), a robust extention of PiCO that is able to mitigate the noisy partial label learning problem.

## Start Running PiCO+

**Run cifar10 with q=0.5 eta=0.2**

```shell
CUDA_VISIBLE_DEVICES=0 python -u train_pico_plus.py \
   --exp-dir experiment/PiCO_plus-CIFAR-10-Noisy --dataset cifar10 --num-class 10\
   --dist-url 'tcp://localhost:10007' --multiprocessing-distributed --world-size 1 --rank 0 --seed 123\
   --arch resnet18 --moco_queue 8192 --prot_start 5 --lr 0.01 --wd 1e-3 --cosine --epochs 800 --print-freq 100\
   --loss_weight 0.5 --proto_m 0.99 --partial_rate 0.5
```

**Run cifar100 with q=0.05 eta=0.2**

```shell
CUDA_VISIBLE_DEVICES=1 python -u train_pico_plus.py \
   --exp-dir experiment/PiCO-CIFAR-100-Noisy --dataset cifar100 --num-class 100\
   --dist-url 'tcp://localhost:10018' --multiprocessing-distributed --world-size 1 --rank 0 --seed 123\
   --arch resnet18 --moco_queue 8192 --prot_start 50 --lr 0.01 --wd 1e-3 --cosine --epochs 800\
   --print-freq 100 --loss_weight 0.5 --proto_m 0.99 --partial_rate 0.05 --chosen_neighbors 5
```
