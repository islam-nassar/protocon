#!/usr/bin/env python
# This code is inspired by Coke from Alibaba group and edited by Islam Nassar
#

import argparse
import builtins
import os
import pickle
import random
import time
import warnings
import math
from collections import OrderedDict
from pathlib import Path
from copy import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models

import builder
import builder.loader
import builder.folder
import builder.optimizer
import builder.builder_protocon
import torch.nn.functional as F
from builder.wideresnet import wide_resnet28w2, wide_resnet28w8
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from builder.loader import DATASETS_STATS
from builder.randaugment import RandAugmentMC


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name])) + ['wide_resnet28w2', 'wide_resnet28w8']
models.__dict__.update({'wide_resnet28w2': wide_resnet28w2, 'wide_resnet28w8': wide_resnet28w8})

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--num_classes', default=10, type=int,
                    help='Number of classes in the dataset.')
parser.add_argument('--x_u_split_file', default=None, type=str,
                    help='pickle file containing a tuple of labeled/unlabelled image names for semi-sup setting.')
parser.add_argument('--num_labels_per_class', default=4, type=int,
                    help='if split file is not passed, this argument is used to obtain labeled/unlabelled image names.')
parser.add_argument('--split_random_seed', default=123, type=int,
                    help='if split file is not passed, this will be the seed for splitting labeled/unlabelled images.')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=64, type=int, metavar='N',
                    help='number of data loading workers (default: 64)')
parser.add_argument('--epochs', default=1001, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size for LABELLED data, this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--mu', default=7, type=int,
                    help='ratio between unlabelled to labelled data in a mini-batch')
parser.add_argument('--optimizer', default='lars', type=str, choices=['sgd', 'lars'],
                    help='Optimizer to use.')
parser.add_argument('--use_amp', action='store_true',
                    help='Use amp autocast or not ')
parser.add_argument('--lr', '--learning-rate', default=1.6, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-6, type=float,
                    metavar='W', help='weight decay (default: 1e-6)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=1, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--log', type=str)
parser.add_argument('--capture_stats', action='store_true',
                    help='If set, unlabelled statistics dataframe will be captured.')
# options for debiasing
parser.add_argument('--refine', action='store_true',
                    help='If set, pseudo-labels are refined with K-means cluster pseudo-labels.')
parser.add_argument('--weighted_cpl', action='store_true',
                    help='If set, cluster pseudo-labels are weighted by sample distances from cluster centroids.')
parser.add_argument('--reliable_cpl', action='store_true',
                    help='If set, cluster pseudo-labels are only calculated based on reliable samples.')
parser.add_argument('--inverse_masking', action='store_true',
                    help='If set, consistency loss will be maksed with the inverse confidence mask.')
parser.add_argument('--dist_align', action='store_true',
                    help='If set, pseudo-labels will be distribution-aligned.')
parser.add_argument('--debias', action='store_true',
                    help='If set, pseudo-labels will be debiased.')
parser.add_argument('--debias_coeff', default=0.5, type=float,
                    help='debiasing coefficient')
parser.add_argument('--mask_pbar', action='store_true', default=False,
                    help='Use the confidence threshold tau to mask pbar')
parser.add_argument('--pbar_gamma', default=0.999, type=float,
                    help='pbar momentum for EMA')
# options for builder
parser.add_argument('--warmup_epochs', default=50, type=int,
                    help='Protocon Warmup Epochs. During which, no consistency losses will be applied. ')
parser.add_argument('--tau', default=0.9, type=float,
                    help='confidence threshold for classifier head.')
parser.add_argument('--tau_proj', default=None, type=float,
                    help='confidence threshold for prototypical head, if None, same mask as classifier is used.')
parser.add_argument('--hist_size', default=2, type=int,
                    help='History size for pseudo-label queue')
parser.add_argument('--consistency_crit', default='strict', type=str,
                    help="""If strict, sample is said to be consistently-reliable if both weak and strong augmentations 
                    have consistent prediction and label is the same across last hist_size epochs e.g. w=[1 1] s=[1 1]. 
                    If other, a sample is consistent if weak and strong have same pseudo-label in each individual epoch 
                    but not neccessarily the same across epochs e.g. w=[1 2] s=[1 2] """)
parser.add_argument('--strong_aug', default='randaugment', type=str, choices=['randaugment', 'simclr'],
                    help='Augmentation used for strong view')
parser.add_argument('--lambda-x', default=1, type=float,
                        help='coefficient of labeled classification loss')
parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss on strong augmentation')
parser.add_argument('--lambda-p-x', default=1, type=float,
                        help='coefficient of prototypical labelled loss')
parser.add_argument('--lambda-p-u', default=1, type=float,
                        help='coefficient of prototypical unlabelled loss')
parser.add_argument('--lambda-c', default=1, type=float,
                        help='coefficient of consistency loss')
parser.add_argument('--mixup_alpha', default=8, type=float,
                        help='alpha parameter for mixup beta distribution')
parser.add_argument('--mixup_strategy', default='mix_mix', type=str, choices=['lab_conf', 'mix_mix', 'mix_non_conf'],
                        help='Mixup strategy')
parser.add_argument('--pseudo_label', default='soft', type=str, choices=['soft', 'hard'],
                        help='if soft, soft x-ent will be applied to Lu')
parser.add_argument('--proto_loss', default='soft-xent', type=str, choices=['soft-xent', 'hinge'],
                        help='prototypical loss type')
parser.add_argument('--consistency_loss', default='instance', type=str, choices=['instance', 'contrastive'],
                        help='consistency loss type')
parser.add_argument('--masked_cluster', action='store_true',
                    help='If set, cluster loss will be masked opposite to pseudo-labels.')
parser.add_argument('--builder-dim', default=64, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--builder-k', default=[3000, 4000, 5000], type=int, nargs="+", help='multi-clustering head')
parser.add_argument('--builder-t', default=0.05, type=float,
                    help='temperature (default: 0.05)')
parser.add_argument('--builder-dual-lr', default=20., type=float,
                    help='dual learning rate')
parser.add_argument('--builder-ratio', default=1.0, type=float,
                    help='ratio of lower-bound')
parser.add_argument('--builder-alpha', default=0.5, type=float,
                    help='weight of classifier pseudo-label. (1 - builder-alpha) is th weight of the cluster pseudo-label')


def main():
    args = parser.parse_args()
    print(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
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
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # Data loading code
    traindir = os.path.join(args.data, 'train')

    aug_weak, aug_strong = get_augmentations(args)

    if args.x_u_split_file:
        labelled_imgs, unlabelled_imgs = pickle.load(Path(args.x_u_split_file).open('rb'))
    else:
        labelled_imgs, unlabelled_imgs = builder.folder.create_nshot_semisup_lists(args.data, args.num_classes,
                                                                                    args.num_labels_per_class,
                                                                                    args.split_random_seed)
    # define additional useful args
    args.proto_num_ins = len(unlabelled_imgs)
    args.proto_num_head = len(args.proto_k)

    print(f'labelled ({len(labelled_imgs)}): {labelled_imgs[:3]},... '
          f'unlabelled ({len(unlabelled_imgs)}): {unlabelled_imgs[:3]}')

    labelled_dataset = builder.folder.SelectiveImageFolder(
        root=traindir,
        selected_imgs=labelled_imgs,
        min_num_samples=len(unlabelled_imgs),  # to avoid multiple stop iterations with the labelled data loader
        transform=builder.loader.SingleCropsTransform(transforms.Compose(aug_weak)))

    unlabelled_dataset = builder.folder.SelectiveImageFolder(
        root=traindir,
        selected_imgs=unlabelled_imgs,
        transform=builder.loader.DoubleCropsTransform(transforms.Compose(aug_weak),
                                                       transforms.Compose(aug_strong)))
    # Initialise ProtoCon
    protocon = ProtoConAlg(args)

    if args.capture_stats:
        protocon.stats['class_mapping'] = copy(unlabelled_dataset.class_to_idx)
        protocon.stats['imgs'] = copy(unlabelled_dataset.imgs)
    if args.distributed:
        labelled_sampler = torch.utils.data.distributed.DistributedSampler(labelled_dataset)
        unlabelled_sampler = torch.utils.data.distributed.DistributedSampler(unlabelled_dataset)
    else:
        labelled_sampler = None
        unlabelled_sampler = None

    # IMPORTANT: Since I reinitialise the labelled iterator many times in an epoch, not setting
    # persistent_workers to true will slow down the training significantly - unless dataset is expanded
    labelled_loader = torch.utils.data.DataLoader(
        labelled_dataset, batch_size=args.batch_size, shuffle=(labelled_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=labelled_sampler, drop_last=True,
        persistent_workers=args.workers > 0)

    unlabelled_loader = torch.utils.data.DataLoader(
        unlabelled_dataset, batch_size=args.batch_size * args.mu, shuffle=(unlabelled_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=unlabelled_sampler, drop_last=True)
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = protocon.builder_protocon.ProtoConModel(
        base_encoder=models.__dict__[args.arch],
        num_classes=args.num_classes,
        dim=args.proto_dim,
    )



    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model) # moved by Islam to here
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            protocon.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            print(f'Batch Size per GPU:{args.batch_size}')
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            print(f'Workers per GPU:{args.workers}')
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        protocon.cuda(args.gpu)
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(grouped_parameters, lr=args.lr,
                              momentum=args.momentum, nesterov=True)
    elif args.optimizer == 'lars':
        optimizer = protocon.optimizer.LARS(grouped_parameters, args.lr,
                                            # weight_decay=args.weight_decay,
                                            momentum=args.momentum)
    else:
        raise NotImplementedError(f"{args.optimizer} optimizer is not supported.")
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
            args.start_epoch = checkpoint['epoch'] if args.start_epoch is None else args.start_epoch
            if args.start_epoch != 0:
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                checkpoint['state_dict'] = OrderedDict({k: v for k, v in checkpoint['state_dict'].items()
                                                        if 'projector' not in k and 'predictor' not in k})
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        args.start_epoch = 0

    cudnn.benchmark = True

    scaler = GradScaler(enabled=args.use_amp)
    flag = True
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            unlabelled_sampler.set_epoch(epoch)

        if epoch == args.start_epoch:
            global labelled_epoch  #  to keep track of labelled images epochs (cause we use 2 data loaders)
            labelled_epoch = 0
            global lab_iterator
            if args.distributed:
                labelled_sampler.set_epoch(labelled_epoch)
            lab_iterator = iter(labelled_loader)

        # train for one epoch
        protocon.train(labelled_loader, unlabelled_loader, model, optimizer, epoch, scaler)
        protocon.reset_cluster_counters()
        protocon.calculate_cluster_pseudo_labels()
        protocon.update_prototypes()
        with torch.no_grad():
            sim = protocon.prototypes.matmul(protocon.prototypes.T).clone().view(1,-1).squeeze(0).cpu().numpy()
            sim = [elem for elem in sorted(list(sim)) if round(elem, 2) != 1]
            if len(sim):
                print(f'Prototypes max similarity:{sim[-1]}, min sim: {sim[0]}')
            else:
                if flag:
                    print(protocon.prototypes)
                    flag = False
                print(f'Prototypes are identical')
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            if epoch % 100 == 0 and epoch:
                torch.save(protocon.prototypes.clone().cpu().numpy(), f'log/prototypes/{args.log}_epoch{epoch}.pth')
            if epoch % 100 == 0 and epoch and args.capture_stats:
                torch.save(protocon.stats, f'log/stats/{args.log}_{epoch}.pth')
                protocon.reset_stats()

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename='model/{}_{:04d}.pth.tar'.format(args.log, epoch))


class ProtoConAlg:
    def __init__(self, args):
        self.args = args
        # ---- builder args
        self.reliable_index = torch.zeros(args.proto_num_ins, dtype=torch.long)
        self.pseudo_label_history = -1 * torch.ones((args.proto_num_ins, args.hist_size), dtype=torch.long)
        self.pseudo_label_history_s = -1 * torch.ones((args.proto_num_ins, args.hist_size), dtype=torch.long)
        self.latest_idx = 0  # to point to the index of freshest pseudo-label in the queue
        self.prototypes = torch.zeros(args.num_classes, args.proto_dim)
        self.prototypes_accum = torch.zeros(args.num_classes, args.proto_dim)
        self.prototypes_counter = torch.zeros(args.num_classes, dtype=torch.long)
        self.pbar = (torch.ones([1, args.num_classes], dtype=torch.float) / args.num_classes)  # used for debiasing the pseudo-labels
        self.pbar_list = []  # used for distribution alignment if applicable
        self.qbar = torch.zeros(1, args.proto_dim)  # used for centering the weak projection before applying contrastive loss
        if args.capture_stats:
            data = {'index': [], 'epoch': [], 'true_class': [], 'max_score_w': [], 'pl_w': [], 'max_score_s': [],
                    'pl_s': [], 'max_score_o': [], 'pl_o': [], 'max_score_c': [], 'pl_c': [], 'max_score_p': [],
                    'pl_p': [], 'clust_id' : [], 'clust_sc' : [], 'loss': []}
            self.stats = {'class_mapping': None, 'imgs': None, 'data': data}
        # ----- clustering args ---
        self.K = args.proto_k
        self.dual_lr = args.proto_dual_lr
        self.ratio = args.proto_ratio
        self.lb = [self.ratio / k for k in self.K]
        self.num_head = args.proto_num_head
        self.cluster_assignment = torch.ones(self.num_head, args.proto_num_ins, dtype=torch.long)
        self.assignment_scores = torch.ones(self.num_head, args.proto_num_ins, dtype=torch.half if args.use_amp else torch.float)
        self.cluster_centers = [F.normalize(torch.randn(args.proto_dim, self.K[i]), dim=0) for i in range(self.num_head)]
        self.duals = [torch.zeros(self.K[i]) for i in range(self.num_head)]
        self.cluster_counters = [torch.zeros(self.K[i]) for i in range(self.num_head)]
        self.cluster_pseudo_labels = [torch.ones(self.K[i], self.args.num_classes) / self.args.num_classes
                                      for i in range(self.num_head)]

    def cuda(self, device):
        self.reliable_index = self.reliable_index.cuda(device)
        self.pseudo_label_history = self.pseudo_label_history.cuda(device)
        self.pseudo_label_history_s = self.pseudo_label_history_s.cuda(device)
        self.prototypes = self.prototypes.cuda(device)
        self.prototypes_accum = self.prototypes_accum.cuda(device)
        self.prototypes_counter = self.prototypes_counter.cuda(device)
        self.pbar = self.pbar.cuda(device)
        self.qbar = self.qbar.cuda(device)
        self.cluster_assignment = self.cluster_assignment.cuda(device)
        self.assignment_scores = self.assignment_scores.cuda(device)
        self.cluster_centers = [self.cluster_centers[i].cuda(device) for i in range(self.num_head)]
        self.duals = [self.duals[i].cuda(device) for i in range(self.num_head)]
        self.cluster_counters = [self.cluster_counters[i].cuda(device) for i in range(self.num_head)]
        self.cluster_pseudo_labels = [self.cluster_pseudo_labels[i].cuda(device) for i in range(self.num_head)]

    def train(self, labelled_loader, unlabelled_loader, model, optimizer, epoch, scaler):
        args = self.args
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        loss_x = AverageMeter('Lx', ':.4e')
        loss_u = AverageMeter('Lu', ':.4e')
        loss_c = AverageMeter('Lc', ':.4e')
        loss_p_x = AverageMeter('Lp_x', ':.4e')
        loss_p_u = AverageMeter('Lp_u', ':.4e')
        losses = AverageMeter('Loss', ':.4e')
        x_top1 = AverageMeter('xAcc@1', ':6.2f')
        pl_top1 = AverageMeter('plAcc@1', ':6.2f')
        pl_orig_top1 = AverageMeter('orig@1', ':6.2f')
        pl_q_top1 = AverageMeter('prot@1', ':6.2f')
        clust_top1 = AverageMeter('clust@1', ':6.2f')
        conf_top1 = AverageMeter('conf@1', ':6.2f')
        cons_top1 = AverageMeter('cons@1', ':6.2f')
        mask_prob = AverageMeter('mask', ':6.2f')
        mask_proj_prob = AverageMeter('mask_p', ':6.2f')
        rel_idx = AverageMeter('rel', ':6.2f')
        pl_cons = AverageMeter('plCons', ':6.2f')

        progress = ProgressMeter(
            len(unlabelled_loader),
            [batch_time, data_time, mask_prob, mask_proj_prob, rel_idx, pl_cons, pl_top1, pl_q_top1, pl_orig_top1, clust_top1,
             conf_top1, cons_top1, loss_x, loss_u, loss_c, loss_p_x, loss_p_u, losses],
            prefix="Epoch: [{}]".format(epoch))

        # switch to train mode
        model.train()
        end = time.time()
        unlab_loader_len = len(unlabelled_loader)
        global lab_iterator
        global labelled_epoch

        for i, (images_u, indexes_u, targets_u) in enumerate(unlabelled_loader):
            adjust_learning_rate(optimizer, epoch, args, i, unlab_loader_len)
            # load labelled data
            try:
                images_x, indexes_x, targets_x = next(lab_iterator)
            except StopIteration:
                if args.distributed:
                    labelled_loader.sampler.set_epoch(labelled_epoch + 1)
                labelled_epoch += 1
                lab_iterator = iter(labelled_loader)
                images_x, indexes_x, targets_x = next(lab_iterator)
            # measure data loading time
            data_time.update(time.time() - end)
            if args.gpu is not None:
                images_u = [view.cuda(args.gpu, non_blocking=True) for view in images_u]  # weak batch and strong batch
                images_x = images_x.cuda(args.gpu, non_blocking=True)
                indexes_u = indexes_u.cuda(args.gpu)
                # indexes_x = indexes_x.cuda(args.gpu)
                targets_u = targets_u.cuda(args.gpu)
                targets_x = targets_x.cuda(args.gpu)

            with autocast(enabled=args.use_amp):
                # ---------------------  prepare mixed up samples and forward pass ---------------------------------
                mu = self.args.mu
                images = interleave(torch.cat([images_x] + images_u, dim=0), 2 * mu + 1)
                classifier_out, projector_out = model(images)
                # ---------------------  deinterleave and separate classifier and projector outputs -----------------
                B = self.args.batch_size
                classifier_out = de_interleave(classifier_out, 2 * mu + 1)
                projector_out = de_interleave(projector_out, 2 * mu + 1)
                # note that p represents logits not probabilities and q is the normalised projection
                p_x, (p_u_w, p_u_s) = classifier_out[:B], classifier_out[B:].chunk(2)
                q_x, (q_u_w, q_u_s) = projector_out[:B], projector_out[B:].chunk(2)
                # ---------------------  apply constrained k-means ----------------------------------------------
                if dist.is_initialized():
                    # to ensure the same cluster center updates happen in all the distributed nodes in the same way
                    q_u_w_all = concat_all_gather(q_u_w)
                    indexes_u_all = concat_all_gather(indexes_u)
                else:
                    q_u_w_all, indexes_u_all = q_u_w, indexes_u
                for head in range(self.num_head):
                    self.update_cluster_assignment(q_u_w_all, indexes_u_all, head)
                    clust_scores, clust_labels = self.get_cluster_assignment(indexes_u_all, head)
                    self.update_cluster_center_mini_batch(q_u_w_all, clust_labels, head)
                #---------------------  calculate builder losses ----------------------------------------------
                # ---------------------  labelled loss-----------------------------
                # labelled loss - Lx (based on true labels and prototypes)
                Lx = F.cross_entropy(p_x, targets_x, reduction='mean')
                logits_proto_x = torch.mm(q_x, self.prototypes.t()) / args.proto_t
                Lp_x = F.cross_entropy(logits_proto_x, targets_x)
                # ---------------------  unlabelled loss with debiasing and refinement -----------------------------
                # unlabelled loss on strong augmentation - Ls
                assert not all([args.debias, args.dist_align]), 'Debiasing and Dist Align can not be both true'
                if args.debias:
                    # debiasing
                    pseudo_label = F.softmax(p_u_w.detach() - args.debias_coeff * torch.log(self.pbar), dim=1)
                elif args.dist_align:
                    # distribution alignment
                    pseudo_label = F.softmax(p_u_w.detach(), dim=1)
                    self.pbar_list.append(pseudo_label.mean(0))
                    if len(self.pbar_list) > 32:  # calculate average based on the last 32 batches only
                        self.pbar_list.pop(0)
                    self.pbar = torch.stack(self.pbar_list, dim=0).mean(0)
                    pseudo_label = pseudo_label / self.pbar
                    # re-normalize to a valid probability
                    pseudo_label = pseudo_label / pseudo_label.sum(dim=1, keepdim=True)
                else:
                    pseudo_label = torch.softmax(p_u_w.detach(), dim=-1)
                # refine pseudo-labels using cluster pseudo-labels
                pseudo_label_orig = pseudo_label.clone()
                clust_pl = self.aggregate_cluster_pseudo_labels(indexes_u)
                if epoch > args.warmup_epochs and args.refine:
                    clust_pl = clust_pl if clust_pl.sum() else pseudo_label  # to handle edge case of epoch 0
                    pseudo_label = args.proto_alpha * pseudo_label + (1 - args.proto_alpha) * clust_pl

                max_probs, targets_u_pl = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(args.tau).float()
                if args.debias:
                    self.pbar = update_pbar(torch.softmax(p_u_w.detach(), dim=-1), self.pbar, momentum=args.pbar_gamma,
                                       pbar_mask=mask if args.mask_pbar else None)
                    p_u_s = p_u_s + args.debias_coeff * torch.log(self.pbar)
                if args.pseudo_label == 'hard':
                    Lu_unreduced = (F.cross_entropy(p_u_s, targets_u_pl, reduction='none'))
                else:
                    Lu_unreduced = torch.sum(-pseudo_label * F.log_softmax(p_u_s, dim=1), dim=1)
                Lu = (Lu_unreduced * mask).mean()
                # Prototypical unlabelled loss
                pseudo_label_q = F.softmax(torch.mm(q_u_w.detach(), self.prototypes.t()) / args.proto_t, dim=1)
                max_score_p, targets_u_proto = torch.max(pseudo_label_q, dim=-1)
                if args.tau_proj is not None:
                    mask_proj = max_score_p.ge(args.tau_proj).float()
                else:
                    mask_proj = mask
                logits_proto_u = torch.mm(q_u_s, self.prototypes.t()) / args.proto_t
                if args.pseudo_label == 'hard':
                    Lp_u = (F.cross_entropy(logits_proto_u, targets_u_pl, reduction='none') * mask_proj).mean()
                else:
                    Lp_u = torch.mean((torch.sum(-pseudo_label * F.log_softmax(logits_proto_u, dim=1), dim=1)) * mask_proj)

                if args.capture_stats:
                    self.update_stats(indexes_u, epoch, targets_u, max_probs, targets_u_pl, p_u_s,
                                      pseudo_label_orig, pseudo_label_q, clust_pl, Lu_unreduced)

                # consistency loss between weak and strong projections - Lc
                if args.consistency_loss == 'instance':
                    q_out = F.softmax((q_u_w.detach() - self.qbar) / (args.proto_t / 5), dim=1)
                    if args.inverse_masking:
                        inv_mask = mask_proj.bool().logical_not().float()
                        Lc = (torch.sum(-q_out * F.log_softmax(q_u_s / args.proto_t, dim=1), dim=1) * inv_mask).mean()
                    else:
                        Lc = torch.mean(torch.sum(-q_out * F.log_softmax(q_u_s / args.proto_t, dim=1), dim=1))
                    self.update_qbar(q_out)
                elif args.consistency_loss == 'contrastive':
                    Lc = instance_contrastive_loss(q_u_w, q_u_s, args.proto_t, return_accuracy=False)
                # total loss
                loss = args.lambda_x * Lx + args.lambda_p_x * Lp_x + args.lambda_u * Lu + args.lambda_p_u * Lp_u + \
                       args.lambda_c * Lc
            # ---------------------  update pseudo-label history and accumulate prototypes-----------------------------
            mask_cons = self.update_history_and_reliability(pseudo_label, p_u_s, indexes_u, epoch)
            self.accumulate_prototypes(q_x, q_u_w, targets_x, indexes_u,
                                       include_labelled=True)
                                       # include_labelled=(i == 0 or i == unlab_loader_len - 1))
            # ---------------------  capture stats ------------------------------------------------------------------
            pl_acc1, _ = accuracy(pseudo_label, targets_u, topk=(1, 5))
            pl_orig_acc1, _ = accuracy(pseudo_label_orig, targets_u, topk=(1, 5))
            pl_q_acc1, _ = accuracy(pseudo_label_q, targets_u, topk=(1, 5))
            clust_acc1, _ = accuracy(clust_pl, targets_u, topk=(1, 5))
            conf_acc1 = (targets_u_pl[mask.bool()] == targets_u[mask.bool()]).float().mean().item()
            cons_acc1 = (targets_u_pl[mask_cons.bool()] == targets_u[mask_cons.bool()]).float().mean().item()
            x_acc1, _ = accuracy(p_x, targets_x, topk=(1, 5))
            pl_top1.update(pl_acc1[0], images_u[0].size(0))
            pl_orig_top1.update(pl_orig_acc1[0], images_u[0].size(0))
            pl_q_top1.update(pl_q_acc1[0], images_u[0].size(0))
            clust_top1.update(clust_acc1[0], images_u[0].size(0))
            if mask.long().sum().item():
                conf_top1.update(conf_acc1)
            if mask_cons.sum().item():
                cons_top1.update(cons_acc1)
            x_top1.update(x_acc1[0], images_x.size(0))
            mask_prob.update(mask.mean().item())
            mask_proj_prob.update(mask_proj.mean().item())
            rel_idx.update(self.reliable_index[indexes_u].float().mean().item())
            pl_cons.update(mask_cons.float().mean().item())
            loss_x.update(Lx.item(), images_x.size(0))
            loss_u.update(Lu.item(), images_u[0].size(0))
            loss_c.update(Lc.item(), images_u[0].size(0))
            loss_p_x.update(Lp_x.item(), images_x.size(0))
            loss_p_u.update(Lp_u.item(), images_x.size(0))
            losses.update(loss.item(), images_u[0].size(0))
            # ---------------------  backward step ---------------------------------
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                progress.display(i)
        progress.display(unlab_loader_len)
        for i in range(0, args.proto_num_head):
            temp = self.cluster_counters[i].data
            print('max and min cluster size for {}-class clustering is ({},{})'.format(args.proto_k[i], torch.max(
                temp).item(), torch.min(temp).item()))

    @torch.no_grad()
    def update_qbar(self, q):
        """
        Update center used for projection.
        """
        batch_center = torch.sum(q, dim=0, keepdim=True)
        if dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center = batch_center / (len(q) * dist.get_world_size())
        else:
            batch_center = batch_center / len(q)
        # ema update
        self.qbar = self.qbar * 0.9 + batch_center * (1 - 0.9)

    @torch.no_grad()
    def aggregate_cluster_pseudo_labels(self, indexes):
        clust_pl = []
        for head in range(self.args.proto_num_head):
            _, clust_u = self.get_cluster_assignment(indexes, head)
            clust_pl.append(self.cluster_pseudo_labels[head][clust_u])
        return torch.stack(clust_pl, dim=0).mean(dim=0)

    @torch.no_grad()
    def update_history_and_reliability(self, pseudo_label, p_u_s, indexes_u, epoch):
        if dist.is_initialized():
            # gather across processes to update in the same way for all processes
            pseudo_label_all = concat_all_gather(pseudo_label)
            p_u_s_all = concat_all_gather(p_u_s)
            indexes_u_all = concat_all_gather(indexes_u)
        else:
            pseudo_label_all, p_u_s_all, indexes_u_all = pseudo_label, p_u_s, indexes_u
        # update history
        scores, pseudo = torch.max(pseudo_label_all, dim=1)  # note that pseudo_label is probs not logits unlike p_u_s
        scores_s, pseudo_s = torch.max(F.softmax(p_u_s_all, dim=1), dim=1)
        self.pseudo_label_history[indexes_u_all, epoch % self.args.hist_size] = pseudo
        self.pseudo_label_history_s[indexes_u_all, epoch % self.args.hist_size] = pseudo_s
        self.latest_idx = epoch % self.args.hist_size
        # calculate consistency mask - whether history of weak and strong pseudo labels are consistent for each image
        mask_cons = (self.pseudo_label_history[indexes_u_all] == self.pseudo_label_history_s[
            indexes_u_all]).float().sum(1) == self.args.hist_size
        if self.args.consistency_crit == 'strict':
            # if strict also check that the label is identical throughout the history
            for i in range(self.args.hist_size - 1):
                mask_cons.logical_and_(
                    self.pseudo_label_history[indexes_u_all, i] == self.pseudo_label_history[
                        indexes_u_all, i + 1])
        # confidence mask
        mask_conf = scores.ge(self.args.tau)
        # update reliability index
        self.reliable_index[indexes_u_all] = mask_conf.long()   #mask_cons.logical_or(mask_conf).long() ## original
        return mask_cons

    @torch.no_grad()
    def accumulate_prototypes(self, feats_x, feats_u, targets_x, indexes_u, include_labelled=False):
        if include_labelled:
            # update prototypes using labelled instances only for first and last iteration of an epoch
            self.prototypes_accum.index_add_(0, targets_x, feats_x)
            counts = torch.ones_like(targets_x)
            self.prototypes_counter.index_add_(0, targets_x, counts)
        # update prototypes using reliable unlabelled samples
        idx = self.latest_idx
        targets_u = self.pseudo_label_history[indexes_u, idx]
        if not (targets_u == -1).sum():
            mask_reliability = self.reliable_index[indexes_u].unsqueeze(1)
            self.prototypes_accum.index_add_(0, targets_u, feats_u * mask_reliability)
            counts = torch.ones_like(targets_u) * mask_reliability.squeeze(1)
            self.prototypes_counter.index_add_(0, targets_u, counts)

    @torch.no_grad()
    def update_prototypes(self):
        if dist.is_initialized():
            dist.all_reduce(self.prototypes_accum)
            dist.all_reduce(self.prototypes_counter)
        self.prototypes = F.normalize(self.prototypes_accum / self.prototypes_counter.unsqueeze(1), dim=1)
        # reset accumulator and counter
        self.prototypes_accum = 0 * self.prototypes_accum
        self.prototypes_counter = 0 * self.prototypes_counter

    @torch.no_grad()
    def update_cluster_assignment(self, feats, indexes, head):
        sim = torch.mm(feats, self.cluster_centers[head])
        scores, labels = torch.max(sim + self.duals[head], dim=1)
        self.cluster_assignment[head][indexes] = labels
        self.assignment_scores[head][indexes] = scores.half() if self.args.use_amp else scores

    @torch.no_grad()
    def get_cluster_assignment(self, indexes, head):
        return self.assignment_scores[head][indexes], self.cluster_assignment[head][indexes]

    @torch.no_grad()
    def reset_cluster_counters(self):
        # happens once after each epoch
        for i in range(0, self.num_head):
            self.cluster_counters[i] = 0 * self.cluster_counters[i]

    @torch.no_grad()
    def calculate_cluster_pseudo_labels(self):
        # happens once after each epoch
        for i in range(0, self.num_head):
            for cluster in range(self.K[i]):
                index = self.cluster_assignment[i] == cluster  # obtain index of samples belonging to a given cluster
                if self.args.reliable_cpl:
                    reliable_index = index.logical_and(self.reliable_index.bool())
                    if (reliable_index.sum() / index.sum()).item() >= 0.9:  # only use reliable index when there is enough samples
                        index = reliable_index
                labels = self.pseudo_label_history[index, self.latest_idx]
                classes, counts = torch.unique(labels, return_counts=True)
                if not (classes == -1).sum():
                    if self.args.weighted_cpl:
                        weights = torch.zeros_like(self.cluster_pseudo_labels[i][cluster])
                        scores = self.assignment_scores[i][index].type(weights.dtype)
                        # weigh each class with overall distance of its samples to the cluster centroid
                        weights.index_add_(0, labels, scores)
                        counts = counts * weights[classes]
                    # obtain cluster pseudo-label as the normalised (or un-normalised) weighted counts
                    temp = torch.zeros_like(self.cluster_pseudo_labels[i][cluster])
                    temp[classes] = counts/counts.sum()
                    self.cluster_pseudo_labels[i][cluster] = temp
            print(f'Number of non-empty clusters in head {i}: {self.cluster_pseudo_labels[i].sum().item()}')


    @torch.no_grad()
    def update_cluster_center_mini_batch(self, feats, clust_labels, head):
        label_idx, label_count = torch.unique(clust_labels, return_counts=True)
        self.duals[head][label_idx] -= self.dual_lr / len(clust_labels) * label_count
        self.duals[head] += self.dual_lr * self.lb[head]
        if self.ratio < 1:
            self.duals[head][self.duals[head] < 0] = 0
        alpha = self.cluster_counters[head][label_idx].float()
        self.cluster_counters[head][label_idx] += label_count
        alpha = alpha / self.cluster_counters[head][label_idx].float()
        self.cluster_centers[head][:, label_idx] = self.cluster_centers[head][:, label_idx] * alpha
        self.cluster_centers[head].index_add_(1, clust_labels, feats.data.T * (1. / self.cluster_counters[head][clust_labels]))
        self.cluster_centers[head][:, label_idx] = F.normalize(self.cluster_centers[head][:, label_idx], dim=0)

    @torch.no_grad()
    def update_stats(self, indexes_u, epoch, targets_u, max_probs, targets_u_pl, p_u_s,
                          pseudo_label_orig, pseudo_label_q, clust_pl, Lu_unreduced):
        self.stats['data']['index'].extend(indexes_u.clone().cpu().numpy())
        self.stats['data']['epoch'].extend([epoch] * indexes_u.size(0))
        self.stats['data']['true_class'].extend(targets_u.clone().cpu().numpy())
        self.stats['data']['max_score_w'].extend(max_probs.clone().cpu().numpy())
        self.stats['data']['pl_w'].extend(targets_u_pl.clone().cpu().numpy())
        max_probs_s, targets_u_pl_s = torch.max(F.softmax(p_u_s, dim=1), dim=1)
        self.stats['data']['max_score_s'].extend(max_probs_s.clone().cpu().numpy())
        self.stats['data']['pl_s'].extend(targets_u_pl_s.clone().cpu().numpy())
        self.stats['data']['loss'].extend(Lu_unreduced.clone().cpu().numpy())
        #----
        sc, tgt = torch.max(pseudo_label_orig, dim=1)
        self.stats['data']['max_score_o'].extend(sc.clone().cpu().numpy())
        self.stats['data']['pl_o'].extend(tgt.clone().cpu().numpy())
        # ----
        sc, tgt = torch.max(pseudo_label_q, dim=1)
        self.stats['data']['max_score_p'].extend(sc.clone().cpu().numpy())
        self.stats['data']['pl_p'].extend(tgt.clone().cpu().numpy())
        # ----
        sc, tgt = torch.max(clust_pl, dim=1)
        self.stats['data']['max_score_c'].extend(sc.clone().cpu().numpy())
        self.stats['data']['pl_c'].extend(tgt.clone().cpu().numpy())
        # ----
        sc, clust_u = self.get_cluster_assignment(indexes_u, 0)
        self.stats['data']['clust_id'].extend(clust_u.clone().cpu().numpy())
        self.stats['data']['clust_sc'].extend(sc.clone().cpu().numpy())

    def reset_stats(self):
        data = {'index': [], 'epoch': [], 'true_class': [], 'max_score_w': [], 'pl_w': [], 'max_score_s': [],
                'pl_s': [], 'max_score_o': [], 'pl_o': [], 'max_score_c': [], 'pl_c': [], 'max_score_p': [],
                'pl_p': [], 'clust_id': [], 'clust_sc': [], 'loss': []}






def save_checkpoint(state, filename='checkpoint.pth.tar', save_freq=1000):
    if (state['epoch'] - 1) % save_freq != 0 or state['epoch'] == 1:
        return
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args, iteration, num_iter):
    warmup_epoch = 11
    warmup_iter = warmup_epoch * num_iter
    current_iter = iteration + epoch * num_iter
    max_iter = args.epochs * num_iter
    lr = args.lr * (1. + math.cos(math.pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
    if epoch < warmup_epoch:
        lr = args.lr * max(1, current_iter - num_iter) / (warmup_iter - num_iter)
    if epoch == 0:
        lr = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_augmentations(args):
    dataset_name = 'cifar100' if 'cifar100' in args.data else 'cifar10' if 'cifar10' in args.data else 'imagenet'
    mean, std = DATASETS_STATS[dataset_name]
    normalize = transforms.Normalize(mean=mean,
                                     std=std)

    aug_1 = [
        transforms.RandomResizedCrop(224, scale=(0.08, 1.)) if 'cifar' not in dataset_name else
        transforms.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode='reflect'),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([builder.loader.GaussianBlur([.1, 2.])], p=1.0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]


    weak = [
        transforms.RandomResizedCrop(224, scale=(0.08, 1.)) if 'cifar' not in dataset_name else
        transforms.RandomCrop(size=32,
                              padding=int(32 * 0.125),
                              padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]
    if args.strong_aug == 'randaugment':
        strong = [
            transforms.RandomResizedCrop(224, scale=(0.08, 1.)) if 'cifar' not in dataset_name else
            transforms.RandomCrop(size=32,
                                  padding=int(32 * 0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    else:
        strong = [
            transforms.RandomResizedCrop(224, scale=(0.08, 1.)) if 'cifar' not in dataset_name else
            transforms.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode='reflect'),
            transforms.RandomApply([
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([builder.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomApply([builder.loader.Solarize()], p=0.3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

    return weak, strong

def update_pbar(probs, pbar, momentum, pbar_mask=None):
    if pbar_mask is not None:
        mean_prob = probs.detach() * pbar_mask.detach().unsqueeze(dim=-1)
    else:
        mean_prob = probs.detach().mean(dim=0)
    pbar = momentum * pbar + (1 - momentum) * mean_prob
    return pbar


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


#adapted from https://github.com/ShaoTengLiu/Hyperbolic_ZSL/
class HingeLoss(torch.nn.Module):
    def __init__(self, margin):
        super(HingeLoss, self).__init__()
        self.margin = margin

    def forward(self, output, target):
        if len(output) <= 1:
            return torch.tensor(0.).cuda()  # zero loss if a single image batch is encountered
        loss = 0
        for i in range(len(output)):
            v_image = output[i]
            t_label = target[i]
            j = np.random.randint(0, len(output)-1)
            while j == i:
                j = np.random.randint(0, len(output)-1)
            t_j = target[j]
            cos = nn.CosineSimilarity(dim=0, eps=1e-6)
            if not torch.allclose(t_j, t_label):
                loss += torch.relu(self.margin - cos(t_label, v_image) + cos(t_j, v_image))
            else:
                loss += torch.relu(-self.margin - cos(t_label, v_image) + cos(t_j, v_image))  # zero loss if the t_j happens to be the same as the true label t_label
        return loss / len(output)

# in progress
class HingeLoss2(torch.nn.Module):

    def __init__(self, margin):
        super(HingeLoss2, self).__init__()
        self.margin = margin

    def forward(self, output, target, target_idx):
        with torch.no_grad():
            # get negative target
            idx = torch.randperm(target.size(0))
            neg = target_idx[idx]
            counter = 0
            while (neg == target_idx).sum() and counter < 10:
                idx = torch.randperm(target.size(0))
                neg = target_idx[idx]
                counter += 1
            mask = (neg != target_idx).float()
            negative_target = target[neg]
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        loss = torch.relu(self.margin - cos(target, output) + cos(negative_target, output)) * mask
        return loss.mean()

def instance_contrastive_loss(feat, feat_aug, temperature, return_accuracy=False):
        ##**************Instance contrastive loss****************
        batch_size = feat.size(0)
        sim_clean = torch.mm(feat, feat.t())
        mask = (torch.ones_like(sim_clean) - torch.eye(batch_size, device=sim_clean.device)).bool()
        sim_clean = sim_clean.masked_select(mask).view(batch_size, -1)

        sim_aug = torch.mm(feat, feat_aug.t())
        sim_aug = sim_aug.masked_select(mask).view(batch_size, -1)

        logits_pos = torch.bmm(feat.view(batch_size, 1, -1), feat_aug.view(batch_size, -1, 1)).squeeze(-1)
        logits_neg = torch.cat([sim_clean, sim_aug], dim=1)

        logits = torch.cat([logits_pos, logits_neg], dim=1)
        instance_labels = torch.zeros(batch_size).long().cuda()

        loss = F.cross_entropy(logits / temperature, instance_labels)
        if return_accuracy:
            acc = accuracy(logits, instance_labels)[0].item()
            return loss, acc
        return loss

if __name__ == '__main__':
    main()
