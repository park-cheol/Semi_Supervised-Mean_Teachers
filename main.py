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
# shutil 모듈은 파일과 파일 모음에 대한 여러 가지 고수준 연산을 제공
# 특히, 파일 복사와 삭제를 지원하는 함수가 제공

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
import torchvision.datasets
import torchvision.utils
import torchvision.transforms as transforms
from torch.autograd import Variable

import dataset
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
parser.add_argument("--BN", default=True, type=bool, help="Use Batch Norm")

# dataset argument
parser.add_argument("--dataset", type=str, default="cifar10",
                    help="dataset name")
parser.add_argument("--datadir", type=str, default="data-local/images/cifar10")
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
                    help="evaluate model on evaluation set")
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# multi processing arguemnt
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
best_prec1 = 0
global_step = 0

def main():
    # todo global?

    args = parser.parse_args()

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()  # node: server(기계)라고 생각

    main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    global global_step
    global best_prec1
    args.gpu = gpu

    traindir = os.path.join(args.datadir, args.train_subdir)
    evaldir = os.path.join(args.datadir, args.eval_subdir)

    checkpoint_path = "saved_models/{}".format(args.dataset)

    model = Net(args)
    ema_model = Net(args)
    # model = ResNet(BottleneckBlock, layers=[4, 4, 4], channels=128, downsample='basic')
    # ema_model = ResNet(BottleneckBlock, layers=[4, 4, 4], channels=128, downsample='basic')


    # teachers model은 학습하는게 아니라 student model에서 weight를 ema해줌
    for param in ema_model.parameters():
        param.detach_()

    # CUDA or 환경세팅 or 초기화
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if not torch.cuda.is_available(): # GPU가 없을 시
        print('using CPU, this will be slow')

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        ema_model = ema_model.cuda(args.gpu)
    else:
        # DataParallel은 사용가능한 gpu에다가 batchsize을 나누고 할당
        model = nn.DataParallel(model).cuda(args.gpu)
        ema_model = nn.DataParallel(ema_model).cuda(args.gpu)

    # optim / loss
    # paper에서 adam이라고 되어있음 ==> 확실히 Adam이 좀더 높게 나옴
    optimizer = torch.optim.Adam(model.parameters(),
                                args.lr,
                                betas=(0.9, 0.999))

    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay,
    #                             nesterov=args.nesterov)

    class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cuda()
    # ignore_index: 특정 라벨을 mask할 때 사용
    # semantic segmentation경우 'dont care'의미하는 -1 label이 있는데 그걸 무시하라는 말

    # 저장된 weight있으면 불러오기
    if args.resume:
        print("loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("loaded checkpoint epoch:", checkpoint['epoch'])

    cudnn.benchmark = True
    # 내장된 cudnn 자동 튜너 활성화, 하드웨어에 맞게 최상의 알고리즘 찾음, 같은 이미지 크기만 들어오실 좋음

    # transform 이미지 불러오기
    train_transform = TransformTwice(transforms.Compose([
        RandomTranslateWithReflect(4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]))

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    train_dataset = torchvision.datasets.ImageFolder(traindir, train_transform)
    eval_dataset = torchvision.datasets.ImageFolder(evaldir, eval_transform)

    if args.labels:
        with open(args.labels) as f:
            labels = dict(line.split(' ') for line in f.read().splitlines())
            # labels 분리 label text 확인
        labeled_idxs, unlabeled_idxs = relabel_dataset(train_dataset, labels)

    if args.labeled_batch_size:
        batch_sampler = TwoStreamBatchSampler(
            unlabeled_idxs, labeled_idxs, args.batch_size, args.labeled_batch_size
        )

    else:
        assert False, "labeled batch size {}".format(args.labeled_batch_size)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_sampler=batch_sampler,
                                               num_workers=args.workers,
                                               pin_memory=True)

    eval_loader = torch.utils.data.DataLoader(eval_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=2 * args.workers, # needs image twice as fast
                                              pin_memory=True,
                                              drop_last=False)
    # drop_last: 마지막 batch는 길이가 실제로 다를 수 있으므로 그걸 배제 할 수 도있음


    # 평가
    if args.evaluate:
        print('Evaluating the student model')
        validate(eval_loader, model, class_criterion, args)
        print('Evaluating the Teacher model')
        validate(eval_loader, ema_model, class_criterion, args)

        return

    print("total 시간check")
    total_start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        train(train_loader, epoch, model, ema_model, class_criterion, optimizer, args)
        print("---epoch time:", datetime.timedelta(seconds=time.time() - start_time))

        if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0: # eval할 간격
            start_time = time.time()
            print("<Evaluating the student model>")
            acc1 = validate(eval_loader, model, class_criterion, args)
            print("<Evaluating the Teacher model>")
            ema_acc1 = validate(eval_loader, ema_model, class_criterion, args)

            print("---val time:", datetime.timedelta(seconds=time.time() - start_time))
            # 최고 모델 저장
            is_best = ema_acc1 > best_prec1
            best_prec1 = max(ema_acc1, best_prec1) # prec1 = acc1 같은말
        else:
            is_best = False

        # checkpoint 저장
        if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'global_step': global_step,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best, checkpoint_path, epoch + 1)

    print("총 시간:", datetime.timedelta(seconds=time.time() - total_start_time))


def train(train_loader, epoch,model, ema_model, criterion, optimizer, args):
    global global_step

    # consistency_type 고르기
    if args.consistency_type == 'mse': # default 되어있음
        consistency_criterion = softmax_mes_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = softmax_kl_loss
    else:
        assert False, args.consistency_type

    meters = AverageMeterSet()

    model.train()
    ema_model.train()

    end = time.time()

    for i, ((input, ema_input), target) in enumerate(train_loader):
        # measure data loading time
        # target: train_loader 자체에서 폴더 수로 class를 분류 해주는 것 같다.(그걸 one-hot vecotr 바꿔줌)
        meters.update('data_time', time.time() - end)

        adjust_learning_rate(optimizer, epoch, i, len(train_loader), args)
        meters.update('lr', optimizer.param_groups[0]['lr'])

        input_var = torch.autograd.Variable(input).cuda(args.gpu, non_blocking=True)
        target_var = torch.autograd.Variable(target.cuda(non_blocking=True))

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        assert labeled_minibatch_size > 0
        meters.update('labeled_minibatch_size', labeled_minibatch_size)

        model_out = model(input_var)

        class_loss = criterion(model_out, target_var) / minibatch_size
        meters.update('class_loss', class_loss.item())

        with torch.no_grad():
            ema_input_var = torch.autograd.Variable(ema_input).cuda(args.gpu, non_blocking=True)
            ema_model_out = ema_model(ema_input_var)

        ema_logit = ema_model_out
        ema_logit = torch.autograd.Variable(ema_logit.detach().data, requires_grad=False)

        if args.consistency:
            consistency_weight = get_current_consistency_weight(epoch, args)
            meters.update('cons_weight', consistency_weight)
            consistency_loss = consistency_weight * consistency_criterion(model_out, ema_logit) / minibatch_size
            meters.update('cons_loss', consistency_loss.item())
        else:
            consistency_loss = 0

        loss = class_loss + consistency_loss
        meters.update('loss', loss.item())

        acc1, acc5 = accuracy(model_out.data, target_var.data, topk=(1, 5))
        meters.update('top1', acc1[0], labeled_minibatch_size)
        meters.update('error1', 100. - acc1[0], labeled_minibatch_size)
        meters.update('top5', acc5[0], labeled_minibatch_size)
        meters.update('error5', 100. - acc5[0], labeled_minibatch_size)

        ema_acc1, ema_acc5 = accuracy(ema_logit.data, target_var.data, topk=(1, 5))
        meters.update('ema_top1', ema_acc1[0], labeled_minibatch_size)
        meters.update('ema_error1', 100. - ema_acc1[0], labeled_minibatch_size)
        meters.update('ema_top5', ema_acc5[0], labeled_minibatch_size)
        meters.update('ema_error5', 100. - ema_acc5[0], labeled_minibatch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1

        update_ema_variables(model, ema_model, args.ema_decay, global_step)

        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print("Epoch: [{}][{}/{}]\tClass_loss {meters[class_loss]:.4f}\tConsistency_loss {meters[cons_loss]:.3f}\tacc@1 {meters[top1]:.3f}\tacc@5 {meters[top5]:.3f}".format(epoch, i, len(train_loader), meters=meters))


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    # paper: alpha는 0에서 시간이 지날수록 점점 증가하는 방법(ramp up)
    # 이러한 방법이 student를 더욱빠르게 초기에 학습
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        # EMA undata 하는 과정
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        # Tensor.add_(a,b) ==> Tensor + (a*b)

# state: 각종 epoch, global_step, state_dict 등
# dirpath: checkpoint_path
def save_checkpoint(state, is_best, dirpath, epoch):
    filename = 'checkpoint.{}.pth'.format(epoch)
    checkpoint_path = os.path.join(dirpath, filename)
    best_path = os.path.join(dirpath, 'best.pth')
    torch.save(state, checkpoint_path)
    print("Save checkpoint ==> (%s)" % filename)

    if is_best:
        shutil.copyfile(checkpoint_path, best_path) # 복사
        print("Save best_accuracy_model_weight ==> (%s)" % filename)



def get_current_consistency_weight(epoch, args):
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)



# epoch: 현재 epoch / step_in_epoch: 현재 iter / total: 총 iter
def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch, args):
    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # large minibatch size handle (lr warm-up)
    lr = linear_rampup(epoch, args.lr_rampup) * (args.lr - args.initial_lr) + args.initial_lr

    # Cosine lr rampdown
    if args.lr_rampdown_epochs:
        assert args.lr_rampdown_epochs >= args.epochs
        lr *= cosine_rampdown(epoch, args.lr_rampdown_epochs)

    for param_group in optimizer.param_groups: # lr 선언해주는 곳
        param_group['lr'] = lr


def validate(eval_loader, model, criterion, args):
    meters = AverageMeterSet()

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(eval_loader):
            meters.update('data_time', time.time() - end)

            input_var = images.cuda(args.gpu, non_blocking=True)
            target_var = target.cuda(args.gpu, non_blocking=True)
            minibatch_size = len(target_var)

            labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
            assert labeled_minibatch_size > 0
            # torch.ne(input, other): input != other 인 것을 True로 반환, 요소들이 같은 경우 False

            # commute output
            output1 = model(input_var)
            class_loss = criterion(output1, target_var) / minibatch_size

            # measure accuracy
            prec1, prec5 = accuracy(output1, target_var.data, topk=(1, 5))
            meters.update('class_loss', class_loss.item(), labeled_minibatch_size)
            meters.update('top1', prec1[0], labeled_minibatch_size)
            meters.update('error1', 100.0 - prec1[0], labeled_minibatch_size)
            meters.update('top5', prec5[0], labeled_minibatch_size)
            meters.update('error5', 100.0 - prec5[0], labeled_minibatch_size)

            meters.update('batch_time', time.time() - end)
            end = time.time()

        print('Acc@1: {meters[top1]:.3f}\tAcc@5: {meters[top5]:.3f}'.format(meters=meters))

    return meters['top1'].avg


# todo 이함수는 나중에 따로
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        labeled_minibatch_size = max(target.ne(NO_LABEL).sum(), 1e-8)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / labeled_minibatch_size))
        return res





if __name__ == '__main__':
    main()






















































































