# Demo shell scripts
# Note that we only tested the single-GPU version
# Please carefully check the code if you would like to use multiple GPUs

# Run cifar10 with q=0.5
CUDA_VISIBLE_DEVICES=0 python -u train.py \
   --exp-dir experiment/PiCO-CIFAR-10 --dataset cifar10 --num-class 10\
   --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --seed 123\
   --arch resnet18 --moco_queue 8192 --prot_start 1 --lr 0.01 --wd 1e-3 --cosine --epochs 800 --print-freq 100\
   --loss_weight 0.5 --proto_m 0.99 --partial_rate 0.5

# Run cifar100 with q=0.05
CUDA_VISIBLE_DEVICES=1 python -u train.py \
   --exp-dir experiment/PiCO-CIFAR-100 --dataset cifar100 --num-class 100\
   --dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0 --seed 123\
   --arch resnet18 --moco_queue 8192 --prot_start 1 --lr 0.01 --wd 1e-3 --cosine --epochs 800 --print-freq 100\
   --loss_weight 0.5 --proto_m 0.99 --partial_rate 0.05

# Run CUB200 with q=0.1
CUDA_VISIBLE_DEVICES=2 python -u train.py \
  --exp-dir experiment/Prot_PLL_CUB --dataset cub200 --num-class 200\
  --dist-url 'tcp://localhost:10003' --multiprocessing-distributed --world-size 1 --rank 0 --seed 124\
  --arch resnet18 --moco_queue 4096 --prot_start 100 --lr 0.01 --wd 1e-5 --cosine --epochs 300 --print-freq 100\
  --batch-size 256 --loss_weight 0.5 --proto_m 0.99 --partial_rate 0.1
