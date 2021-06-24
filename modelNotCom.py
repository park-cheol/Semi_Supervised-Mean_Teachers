######################
# todo official github 미완성
######################

import math
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable, Function

class BottleneckBlock(nn.Module):
    @classmethod
    def out_channels(cls, channels, groups): # todo out_channels?
        if groups > 1:
            return 2 * channels
        else:
            return 4 * channels

    def __init__(self, in_channels, out_channels, groups, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(out_channels)

        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, groups=groups)
        self.bn_2 = nn.BatchNorm2d(out_channels)

        self.conv_3 = nn.Conv2d(out_channels, self.out_channels(out_channels, groups), kernel_size=1, bias=False)
        self.bn_3 = nn.BatchNorm2d(self.out_channels(out_channels, groups))

    def forward(self, input):
        conv_1 = self.conv_1(input)
        bn_1 = self.bn_1(conv_1)
        relu_1 = self.relu(bn_1)

        conv_2 = self.conv2(relu_1)
        bn_2 = self.bn_2(conv_2)
        relu_2 = self.relu(bn_2)

        conv_3 = self.conv_3(relu_2)
        bn_3 = self.bn_3(conv_3)

        if self.downsample is not None:
            input = self.downsample(input)

        return self.relu(input + bn_3)

#######################
# shake regularization
######################
class Shake(Function):
    @classmethod
    def forward(cls, ctx, input1, input2, training):
        assert input1.size() == input2.size()
        gate_size = [input1.size()[0], *itertools.repeat(1, input1.dim() - 1)]
        # [batch, 1, 1, 1]
        gate = input1.new(*gate_size)
        # 같은 type의 새로운 tensor 생성

        # paper: traing일때는 랜덤노이즈 reference일때는 even=0.5로 함
        if training:
            gate.uniform_(0, 1)
        else:
            gate.fill_(0.5)

        return input1 * gate + input2 * (1 - gate)

    @classmethod
    def backward(cls, ctx, grad_output):
        grad_input1 = grad_input2 = grad_training = None
        gate_size = [grad_output.size()[0], *itertools.repeat(1, grad_output.dim() - 1)]

        gate = Variable(grad_output.data.new(*gate_size).unifrom_(0, 1))
        if ctx.needs_input_grad[0]:
            grad_input1 = grad_output * gate
        if ctx.needs_input_grad[1]:
            grad_input2 = grad_output * (1 - gate)

        assert not ctx.needs_input_grad[2]
        return grad_input1, grad_input2, grad_training


class ShakeShakeBlock(nn.Module):
    @classmethod
    def out_channels(cls, out_channels, groups):
        assert groups == 1
        return out_channels

    def __init__(self, in_channels, out_channels, groups, stride=1, downsample=None):
        super(ShakeShakeBlock, self).__init__()
        assert groups == 1
        self.conv_a1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_a1 = nn.BatchNorm2d(out_channels)
        self.conv_a2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_a2 = nn.BatchNorm2d(out_channels)

        self.conv_b1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_b1 = nn.BatchNorm2d(out_channels)
        self.conv_b2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_b2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.stride = stride



class ResNet32x32(nn.Module):
    def __init__(self, blo):