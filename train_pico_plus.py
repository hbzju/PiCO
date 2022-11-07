import argparse
import builtins
import math
import os
import random
from select import select
import shutil
import time
import warnings
import torch
import torch.nn 
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import tensorboard_logger as tb_logger
import numpy as np
from model import PiCO_PLUS
from resnet import *
from utils_plus.utils_algo import *
from utils_plus.utils_loss import partial_loss, SupConLoss, ce_loss
from utils_plus.cub200 import load_cub200
from utils_plus.cifar10 import load_cifar10
from utils_plus.cifar100 import load_cifar100
import copy

torch.autograd.set_detect_anomaly(True)

torch.set_printoptions(precision=2, sci_mode=False)

parser = argparse.ArgumentParser(description='PyTorch implementation of ICLR 2022 Oral paper PiCO')
parser.add_argument('--dataset', default='cifar10', type=str, 
                    choices=['cifar10', 'cifar100', 'cub200'],
                    help='dataset name (cifar10)')
parser.add_argument('--exp-dir', default='experiment/PiCO', type=str,
                    help='experiment directory for saving checkpoints and logs')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', choices=['resnet18'],
                    help='network architecture (only resnet18 used in PiCO)')
parser.add_argument('-j', '--workers', default=32, type=int,
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=500, type=int, 
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, 
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.02, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-lr_decay_epochs', type=str, default='700,800,900',
                    help='where to decay lr, can be a list')
parser.add_argument('-lr_decay_rate', type=float, default=0.1,
                    help='decay rate for learning rate')
parser.add_argument('--cosine', action='store_true', default=False,
                    help='use cosine lr schedule')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:10002', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--num-class', default=10, type=int,
                    help='number of class')
parser.add_argument('--low-dim', default=128, type=int,
                    help='embedding dimension')
parser.add_argument('--moco_queue', default=8192, type=int, 
                    help='queue size; number of negative samples')
parser.add_argument('--moco_m', default=0.999, type=float,
                    help='momentum for updating momentum encoder')
parser.add_argument('--proto_m', default=0.99, type=float,
                    help='momentum for computing the momving average of prototypes')
parser.add_argument('--loss_weight', default=0.5, type=float,
                    help='contrastive loss weight')
parser.add_argument('--conf_ema_range', default='0.95,0.8', type=str,
                    help='pseudo target updating coefficient (phi)')
parser.add_argument('--prot_start', default=80, type=int, 
                    help = 'Start Prototype Updating')
parser.add_argument('--partial_rate', default=0.1, type=float, 
                    help='ambiguity level (q)')
parser.add_argument('--noisy_rate', default=0.2, type=float, 
                    help='noisy level')
parser.add_argument('--hierarchical', action='store_true', 
                    help='for CIFAR-100 fine-grained training')
parser.add_argument('--pure_ratio', default='0.6', type=float,
                    help='selection ratio')
parser.add_argument('--knn_start', default=100, type=int,
                    help='when to start kNN')
parser.add_argument('--chosen_neighbors', default=16, type=int, 
                    help='chosen neighbors')
parser.add_argument('--temperature_guess', default=0.07, type=float, 
                    help='temperature for label guessing')
parser.add_argument('--ur_weight', default='0.1', type=float,
                    help='weights for the losses of unreliable examples')
parser.add_argument('--cls_weight', default=2, type=float,
                    help='weights for the losses of mixup loss')

def main():
    args = parser.parse_args()
    args.conf_ema_range = [float(item) for item in args.conf_ema_range.split(',')]
    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    print(args)

    if args.seed is not None:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    
    model_path = 'ds{ds}p{pr}n{nr}_ps{ps}_lw{lw}_pm{pm}_he_{heir}_sel{sel}_k{k}s{ks}_uw{uw}_sd{seed}'.format(
                                            ds=args.dataset,
                                            pr=args.partial_rate,
                                            ps=args.prot_start,
                                            lw=args.loss_weight,
                                            pm=args.proto_m,
                                            seed=args.seed,
                                            sel=args.pure_ratio,
                                            k=args.chosen_neighbors,
                                            ks=args.knn_start,
                                            nr=args.noisy_rate,
                                            uw=args.ur_weight,
                                            heir=args.hierarchical)
    args.exp_dir = os.path.join(args.exp_dir, model_path)
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    cudnn.benchmark = True
    args.gpu = gpu
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = PiCO_PLUS(args, SupConResNet)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    
    # set optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    if args.dataset == 'cub200':
        input_size = 224  # fixed as 224
        train_loader, train_givenY, train_sampler, test_loader = load_cub200(input_size=input_size
            , partial_rate=args.partial_rate, noisy_rate=args.noisy_rate, batch_size=args.batch_size)
    elif args.dataset == 'cifar10':
        train_loader, train_givenY, train_sampler, test_loader = load_cifar10(partial_rate=args.partial_rate, batch_size=args.batch_size, noisy_rate=args.noisy_rate)
    elif args.dataset == 'cifar100':
        train_loader, train_givenY, train_sampler, test_loader = load_cifar100(partial_rate=args.partial_rate, batch_size=args.batch_size, hierarchical=args.hierarchical, noisy_rate=args.noisy_rate)
    else:
        raise NotImplementedError("You have chosen an unsupported dataset. Please check and try again.")
    # this train loader is the partial label training loader

    print('Calculating uniform targets...')
    num_instance = train_givenY.shape[0]
    tempY = train_givenY.sum(dim=1).unsqueeze(1).repeat(1, train_givenY.shape[1])
    confidence = train_givenY.float()/tempY
    confidence = confidence.cuda()
    # calculate confidence

    loss_fn = partial_loss(confidence)
    loss_cont_fn = SupConLoss()
    # set loss functions (with pseudo-targets maintained)

    if args.gpu==0:
        logger = tb_logger.Logger(logdir=os.path.join(args.exp_dir,'tensorboard'), flush_secs=2)
    else:
        logger = None

    print('\nStart Training\n')

    best_acc = 0
    mmc = 0 #mean max confidence
    sel_stats = {
        'dist': torch.zeros(num_instance).cuda(),
        'is_rel': torch.ones(num_instance).bool().cuda(),
    }

    for epoch in range(args.start_epoch, args.epochs):
        is_best = False
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        adjust_learning_rate(args, optimizer, epoch)
        if epoch >= args.prot_start:
            reliable_set_selection(args, epoch, sel_stats)
            # warm-up for 5 epochs and then start selection
        train(args, train_loader, model, loss_fn, loss_cont_fn, optimizer, epoch, logger, sel_stats)
        loss_fn.set_conf_ema_m(args, epoch)
        # reset phi

        acc_test = test(args, model, test_loader, epoch, logger)
        mmc = loss_fn.confidence.max(dim=1)[0].mean()
        
        with open(os.path.join(args.exp_dir, 'result.log'), 'a+') as f:
            f.write('Epoch {}: Acc {}, Best Acc {}. (lr {}, MMC {})\n'.format(epoch
                , acc_test, best_acc, optimizer.param_groups[0]['lr'], mmc))
        if acc_test > best_acc:
            best_acc = acc_test
            is_best = True

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=is_best, filename='{}/checkpoint.pth.tar'.format(args.exp_dir),
            best_file_name='{}/checkpoint_best.pth.tar'.format(args.exp_dir))

def train(args, train_loader, model, loss_fn, loss_cont_fn, optimizer, epoch, tb_logger, sel_stats):
    batch_time = AverageMeter('Time', ':1.2f')
    data_time = AverageMeter('Data', ':1.2f')
    acc_cls = AverageMeter('Acc@Cls', ':2.2f')
    acc_proto = AverageMeter('Acc@Proto', ':2.2f')
    loss_cls_log = AverageMeter('Loss@Cls', ':2.2f')
    loss_cont_log = AverageMeter('Loss@Cont', ':2.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, acc_cls, acc_proto, loss_cls_log, loss_cont_log],
        prefix="Epoch: [{}]".format(epoch))
    
    start_upd_prot = epoch >= args.prot_start

    # switch to train mode
    model.train()
    
    end = time.time()
    for i, (images_w, images_s, labels, true_labels, index) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        X_w, X_s, Y, index = images_w.cuda(), images_s.cuda(), labels.cuda(), index.cuda()
        Y_true = true_labels.long().detach().cuda()
        # for showing training accuracy and will not be used when training
        is_rel = sel_stats['is_rel'][index]
        batch_weight = is_rel.float()

        cls_out, features_cont, pseudo_labels, score_prot, distance_prot, is_rel_queue\
            = model(X_w, X_s, Y, Y_cor=None, is_rel=is_rel, args=args)
        batch_size = cls_out.shape[0]

        pseudo_target_cont = pseudo_labels.contiguous().view(-1, 1)

        if start_upd_prot:
            loss_fn.confidence_update(temp_un_conf=score_prot, batch_index=index, batchY=Y)
            # warm up ended
            mask_all = torch.eq(pseudo_target_cont[:batch_size], pseudo_target_cont.T).float().cuda()
            loss_cont_all = loss_cont_fn(features=features_cont, mask=mask_all, batch_size=batch_size, weights=None)
            # This is a relaxed version and should be assigned a lower weight

            mask = copy.deepcopy(mask_all).detach()
            mask = batch_weight.unsqueeze(1).repeat(1, mask.shape[1]) * mask
            # remove row-wise unreliable masks
            mask = is_rel_queue.view(1, -1).repeat(mask.shape[0], 1) * mask
            # remove column-wise unreliable masks

            # get positive set by contrasting predicted labels
            if epoch >= args.knn_start:
                cosine_corr = features_cont[:batch_size] @ features_cont.T
                _, kNN_index = torch.topk(cosine_corr, k=args.chosen_neighbors, dim=-1, largest=True)
                # largest cosine correlation indicates closer l2 distance
                mask_kNN = torch.scatter(torch.zeros(mask.shape).cuda(), 1, kNN_index, 1)
                # above: set remaining masks by kNN
                mask[~is_rel] = mask_kNN[~is_rel]

            mask[:, batch_size:batch_size*2] = ((mask[:, batch_size:batch_size*2] + torch.eye(batch_size).cuda())>0).float()
            mask[:, :batch_size] = ((mask[:, :batch_size] + torch.eye(batch_size).cuda())>0).float()
            # reset query/key positivities

            if epoch >= args.knn_start:
                weights = args.loss_weight * batch_weight + args.ur_weight * (1 - batch_weight)
                # for clean data, we use the original loss weight
                # for unreliable data, we calculate a knn-cont loss with a lower weight
                loss_cont_rel_knn = loss_cont_fn(features=features_cont, mask=mask, batch_size=batch_size, weights=weights)
                # jointly calculate the clean/knn-based contrastive loss, but assign knn loss a lower weight
            else:
                loss_cont_rel_knn = loss_cont_fn(features=features_cont, mask=mask, batch_size=batch_size, weights=None)
            # above: contrastive loss on reliable examples

            loss_cont = loss_cont_rel_knn + args.ur_weight * loss_cont_all

            # classification loss
            loss_cls = loss_fn(cls_out, index, is_rel)

            sp_temp_scale = score_prot**(1/args.temperature_guess)
            targets_guess = sp_temp_scale / sp_temp_scale.sum(dim=1, keepdim=True)
            _, loss_cls_ur = ce_loss(cls_out, targets_guess, sel=~is_rel)
            # label guessing on unreliable examples

            l = np.random.beta(4, 4)
            l = max(l, 1-l)
            pseudo_label = loss_fn.confidence[index]
            pseudo_label[~is_rel] = targets_guess[~is_rel]
            # cancat clean label and guessed noisy labels
            idx = torch.randperm(X_w.size(0))
            X_w_rand = X_w[idx]
            pseudo_label_rand = pseudo_label[idx]
            X_w_mix = l * X_w + (1 - l) * X_w_rand      
            pseudo_label_mix = l * pseudo_label + (1 - l) * pseudo_label_rand
            logits_mix, _ = model.module.encoder_q(X_w_mix)
            # use encoder q to avoid DDP error
            _, loss_mix = ce_loss(logits_mix, targets=pseudo_label_mix)
            # mixup loss

            loss_cls = loss_mix + args.cls_weight * loss_cls + args.ur_weight * loss_cls_ur
            # we use the loss_mix as the anchor because it uses all data examples
            loss = loss_cls + loss_cont
        else:
            loss_cls = loss_fn(cls_out, index, is_rel=None)
            loss_cont = loss_cont_fn(features=features_cont, mask=None, batch_size=batch_size)
            # Warmup using MoCo

            loss = loss_cls + args.loss_weight * loss_cont
        # final loss

        sel_stats['dist'][index] = copy.deepcopy(distance_prot.clone().detach())
        # update the distances for data selection

        loss_cls_log.update(loss_cls.item())
        loss_cont_log.update(loss_cont.item())

        # log accuracy
        acc = accuracy(cls_out, Y_true)[0]
        acc_cls.update(acc[0])
        acc = accuracy(score_prot, Y_true)[0] 
        acc_proto.update(acc[0])
 
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time

        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)

    if args.gpu == 0:
        tb_logger.log_value('Train Acc', acc_cls.avg, epoch)
        tb_logger.log_value('Prototype Acc', acc_proto.avg, epoch)
        tb_logger.log_value('Classification Loss', loss_cls_log.avg, epoch)
        tb_logger.log_value('Contrastive Loss', loss_cont_log.avg, epoch)

def reliable_set_selection(args, epoch, sel_stats):
    dist = sel_stats['dist']
    n = dist.shape[0]
    is_rel = torch.zeros(n).bool().cuda()
    sorted_idx = torch.argsort(dist)
    chosen_num = int(n * args.pure_ratio)
    is_rel[sorted_idx[:chosen_num]] = True
    sel_stats['is_rel'] = is_rel
    # select near-prototype samples as reliable and clean

def test(args, model, test_loader, epoch, tb_logger):
    with torch.no_grad():
        print('==> Evaluation...')       
        model.eval()    
        top1_acc = AverageMeter("Top1")
        top5_acc = AverageMeter("Top5")
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images, args, eval_only=True)    
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            top1_acc.update(acc1[0])
            top5_acc.update(acc5[0])
        
        # average across all processes
        acc_tensors = torch.Tensor([top1_acc.avg,top5_acc.avg]).cuda(args.gpu)
        dist.all_reduce(acc_tensors)        
        acc_tensors /= args.world_size
        
        print('Accuracy is %.2f%% (%.2f%%)'%(acc_tensors[0],acc_tensors[1]))
        if args.gpu ==0:
            tb_logger.log_value('Top1 Acc', acc_tensors[0], epoch)
            tb_logger.log_value('Top5 Acc', acc_tensors[1], epoch)             
    return acc_tensors[0]
    
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_file_name='model_best.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_file_name)

if __name__ == '__main__':
    main()
