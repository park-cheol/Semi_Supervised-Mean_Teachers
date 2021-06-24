import argparse
import warnings
import os
import random
import numpy as np
import math
import itertools
import datetime
import time
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

import torchvision
import torchvision.utils
import torchvision.transforms as transforms
from torch.autograd import Variable

from dataset import *
from models import *
from losses import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200, metavar='N',
                    help="number of total epochs of training")
parser.add_argument("--start-epoch", type=int, default=0, metavar='N',
                    help="epochs start training from")
parser.add_argument("--batch-size", type=int, default=256, metavar='N',
                    help="size of the batches")
parser.add_argument("-j", "--workers", type=int, default=4,
                    help="number of data loading workers")
parser.add_argument("--labeled-batch-size", default=62, type=int,
                    help="labeled data per minibatch")

# dataset argument
parser.add_argument("--dataset", type=str, default="cifar10",
                    help="dataset name")
parser.add_argument('--train-subdir', type=str, default='train',
                    help='the subdirectory inside the data directory that contains the training data')
parser.add_argument('--eval-subdir', type=str, default='val',
                    help='the subdirectory inside the data directory that contains the evaluation data')
parser.add_argument('--labels', default='data-local/labels/cifar10/4000_balanced_labels/00.txt', type=str,
                    help="list of image labels(files)")

# learning rate argument
parser.add_argument('--lr', type=int, default=0.2,
                    help="max learning rate")
parser.add_argument('--initial-lr', default=0.0, type=float,
                    help="linear rampup을 사용 할 때 쓸 초기 lr")
parser.add_argument('--lr-rampup', default=0, type=int,
                    help="초반에 lr rampup의 length, epoch")
parser.add_argument('--lr-rampdown-epochs', default=None, type=int,
                    help="cosin_rampdown 사용 시 lr의 길이,epoch(>= training of length)")

# optim argument
parser.add_argument('--momentum', default=0.5, type=float)
parser.add_argument('--nesterov', default=False, type=bool)
parser.add_argument('--weight-decay', default=1e-4, type=float)
parser.add_argument('--ema-decay', default=0.999, type=float,
                    help='ema variable decay rate')

# consistency argument
parser.add_argument('--consistency', default=None, type=float,
                    help='use consistency loss with given weight')
parser.add_argument('--consistency-type', default='mse', type=str,
                    help='mse or kl type')
parser.add_argument('--consistency-rampup', default=30, type=int,
                    help="consistency loss ramp-up epoch")

parser.add_argument('--logit-distance-cost', default=-1, type=float,
                    help="let the student model have two outputs and use an MSE loss between the logits with the given weight(default: only have one output)")
parser.add_argument('--checkpoint-epochs', type=int, default=1,
                    help="저장간격 epoch")
parser.add_argument('--evaluation-epochs', type=int, default=1,
                    help="evaluation 간격 epoch")
parser.add_argument('--print-freq', default=10, type=int,
                    help="print log")
parser.add_argument('--evaluate', type=bool,
                    hep="evaluate model on evaluation set")
# multi processing arguemnt
parser.add_argument("--world-size", default=-1, type=int,
                    help='number of nodes for distributed training ')
parser.add_argument("--rank", default=-1, type=int,
                    help='node rank for distributed training ')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training ')

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
best_prec1 = 0
global_step = 0
























































































