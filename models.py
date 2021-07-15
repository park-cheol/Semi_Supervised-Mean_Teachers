import math
import sys
import itertools


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable, Function

#################
# Noise
#################

class GassianNoise(nn.Module):

    def __init__(self, args, shape=(100, 1, 28, 28), std=0.05):
        super(GassianNoise, self).__init__()
        self.noise1 = Variable(torch.zeros(shape).cuda(args.gpu))
        self.std1 = std
        self.register_buffer('noise2', self.noise1)

    def forward(self, input):
        c = input.shape[0] # batch size
        self.noise2.data.data.normal_(0, std=self.std1)

        return input + self.noise2[:c] # [batch, 3, 32, 32] 초기 채널3
################
# model
################

class Net(nn.Module):

    def __init__(self, args, std = 0.15):
        super(Net, self).__init__()
        self.args = args
        self.std = std

        self.gn = GassianNoise(args, shape=(args.batch_size, 3, 32, 32), std=self.std)

        # 1,2,3 = pooling전까지 block? , a, b, c = block안에서 순서
        if self.args.BN: # batch norm 사용여부
            self.BN1a = nn.BatchNorm2d(128)
            self.BN1b = nn.BatchNorm2d(128)
            self.BN1c = nn.BatchNorm2d(128)

            self.BN2a = nn.BatchNorm2d(256)
            self.BN2b = nn.BatchNorm2d(256)
            self.BN2c = nn.BatchNorm2d(256)

            self.BN3a = nn.BatchNorm2d(512)
            self.BN3b = nn.BatchNorm2d(256)
            self.BN3c = nn.BatchNorm2d(128)

            self.BNdense = nn.BatchNorm1d(100)

        self.conv1a = nn.Conv2d(3, 128, 3, padding=1)
        self.conv1b = nn.Conv2d(128, 128, 3, padding=1)
        self.conv1c = nn.Conv2d(128, 128, 3, padding=1)

        self.conv2a = nn.Conv2d(128, 256, 3, padding=1)
        self.conv2b = nn.Conv2d(256, 256, 3, padding=1)
        self.conv2c = nn.Conv2d(256, 256, 3, padding=1)

        self.conv3a = nn.Conv2d(256, 512, 3)
        self.conv3b = nn.Conv2d(512, 256, 1)
        self.conv3c = nn.Conv2d(256, 128, 1)

        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.AvgPool2d(6, 6) # 1x1 pixels

        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.5)

        self.dense = nn.Linear(128, 100)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels # w * d * h 부피
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, input):
        if self.training: # todo self.training을 지정?
            input = self.gn(input)

        if self.args.BN:
            layer_1a = F.leaky_relu(self.BN1a(self.conv1a(input)), negative_slope=0.1)  # self.BN1a
            layer_1b = F.leaky_relu(self.BN1b(self.conv1b(layer_1a)), negative_slope=0.1)  # self.BN1b
            layer_1c = F.leaky_relu(self.BN1c(self.conv1c(layer_1b)), negative_slope=0.1)  # self.BN1c
            drop_pool_1 = self.drop1(self.pool1(layer_1c))

            layer_2a = F.leaky_relu(self.BN2a(self.conv2a(drop_pool_1)), negative_slope=0.1)  # self.BN2a
            layer_2b = F.leaky_relu(self.BN2b(self.conv2b(layer_2a)), negative_slope=0.1)  # self.BN2b
            layer_2c = F.leaky_relu(self.BN2c(self.conv2c(layer_2b)), negative_slope=0.1)  # self.BN2c
            drop_pool_2 = self.drop2(self.pool2(layer_2c))

            layer_3a = F.leaky_relu(self.BN3a(self.conv3a(drop_pool_2)), negative_slope=0.1)  # self.BN3a
            layer_3b = F.leaky_relu(self.BN3b(self.conv3b(layer_3a)), negative_slope=0.1)  # self.BN3b
            layer_3c = F.leaky_relu(self.BN3c(self.conv3c(layer_3b)), negative_slope=0.1)  # self.BN3c
            pool_3 = self.pool3(layer_3c) # [batch, 128, 1, 1]

            reshaped = pool_3.view(-1, 128)
            dense = self.BNdense(self.dense(reshaped))

        else:
            layer_1a = F.leaky_relu(self.conv1a(input), negative_slope=0.1)
            layer_1b = F.leaky_relu(self.conv1b(layer_1a), negative_slope=0.1)
            layer_1c = F.leaky_relu(self.conv1c(layer_1b), negative_slope=0.1)
            drop_pool_1 = self.drop1(self.pool1(layer_1c))

            layer_2a = F.leaky_relu(self.conv2a(drop_pool_1), negative_slope=0.1)
            layer_2b = F.leaky_relu(self.conv2b(layer_2a), negative_slope=0.1)
            layer_2c = F.leaky_relu(self.conv2c(layer_2b), negative_slope=0.1)
            drop_pool_2 = self.drop2(self.pool2(layer_2c))

            layer_3a = F.leaky_relu(self.conv3a(drop_pool_2), negative_slope=0.1)
            layer_3b = F.leaky_relu(self.conv3b(layer_3a), negative_slope=0.1)
            layer_3c = F.leaky_relu(self.conv3c(layer_3b), negative_slope=0.1)
            pool_3 = self.pool3(layer_3c)

            reshaped = pool_3.view(-1, 128)
            dense = self.dense(reshaped)

        return dense # [128, 10]

##################################
# VGG 16layer
##################################
class VGGNet(nn.Module):

    def __init__(self, args, std = 0.15):
        super(VGGNet, self).__init__()
        self.args = args
        self.std = std

        self.gn = GassianNoise(args, shape=(args.batch_size, 3, 32, 32), std=self.std)

        # feature extractor
        # [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]
        self.conv1_a = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_b = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv2_a = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_b = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.conv3_a = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_b = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_c = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv4_a = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_b = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_c = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv5_a = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_b = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_c = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.bn1_a = nn.BatchNorm2d(64)
        self.bn1_b = nn.BatchNorm2d(64)

        self.bn2_a = nn.BatchNorm2d(128)
        self.bn2_b = nn.BatchNorm2d(128)

        self.bn3_a = nn.BatchNorm2d(256)
        self.bn3_b = nn.BatchNorm2d(256)
        self.bn3_c = nn.BatchNorm2d(256)

        self.bn4_a = nn.BatchNorm2d(512)
        self.bn4_b = nn.BatchNorm2d(512)
        self.bn4_c = nn.BatchNorm2d(512)

        self.bn5_a = nn.BatchNorm2d(512)
        self.bn5_b = nn.BatchNorm2d(512)
        self.bn5_c = nn.BatchNorm2d(512)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.5)
        self.drop3 = nn.Dropout(0.5)
        self.drop4 = nn.Dropout(0.5)
        self.drop5 = nn.Dropout(0.5)
        # Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(4096, 100)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels # w * d * h 부피
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, input):
        input = self.gn(input)
        conv1_a = F.relu(self.bn1_a(self.conv1_a(input)))
        conv1_b = F.relu(self.bn1_b(self.conv1_b(conv1_a)))
        maxpool1 = self.drop1(self.maxpool1(conv1_b))

        conv2_a = F.relu((self.bn2_a(self.conv2_a(maxpool1))))
        conv2_b = F.relu((self.bn2_b(self.conv2_b(conv2_a))))
        maxpool2 = self.drop2(self.maxpool2(conv2_b))

        conv3_a = F.relu((self.bn3_a(self.conv3_a(maxpool2))))
        conv3_b = F.relu((self.bn3_b(self.conv3_b(conv3_a))))
        conv3_c = F.relu((self.bn3_c(self.conv3_c(conv3_b))))
        maxpool3 = self.drop3(self.maxpool3(conv3_c))

        conv4_a = F.relu((self.bn4_a(self.conv4_a(maxpool3))))
        conv4_b = F.relu((self.bn4_b(self.conv4_b(conv4_a))))
        conv4_c = F.relu((self.bn4_c(self.conv4_c(conv4_b))))
        maxpool4 = self.drop4(self.maxpool4(conv4_c))

        conv5_a = F.relu((self.bn5_a(self.conv5_a(maxpool4))))
        conv5_b = F.relu((self.bn5_b(self.conv5_b(conv5_a))))
        conv5_c = F.relu((self.bn5_c(self.conv5_c(conv5_b))))
        maxpool5 = self.drop5(self.maxpool5(conv5_c))

        avgpool = self.avgpool(maxpool5)

        reshaped = torch.flatten(avgpool, 1)
        predict = self.classifier(reshaped)

        return predict








###################################
# RESNET MODEL
###################################
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ShakeShakeBlock(nn.Module): # todo
    @classmethod
    def out_channels(cls, planes, groups):
        assert groups == 1
        return planes

    def __init__(self, inplanes, planes, groups, stride=1, downsample=None):
        super().__init__()
        assert groups == 1
        self.conv_a1 = conv3x3(inplanes, planes, stride)
        self.bn_a1 = nn.BatchNorm2d(planes)
        self.conv_a2 = conv3x3(planes, planes)
        self.bn_a2 = nn.BatchNorm2d(planes)

        self.conv_b1 = conv3x3(inplanes, planes, stride)
        self.bn_b1 = nn.BatchNorm2d(planes)
        self.conv_b2 = conv3x3(planes, planes)
        self.bn_b2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        a, b, residual = x, x, x

        a = F.relu(a, inplace=False)
        a = self.conv_a1(a)
        a = self.bn_a1(a)
        a = F.relu(a, inplace=True)
        a = self.conv_a2(a)
        a = self.bn_a2(a)

        b = F.relu(b, inplace=False)
        b = self.conv_b1(b)
        b = self.bn_b1(b)
        b = F.relu(b, inplace=True)
        b = self.conv_b2(b)
        b = self.bn_b2(b)

        ab = shake(a, b, training=self.training)

        if self.downsample is not None:
            residual = self.downsample(x)

        return residual + ab


class Shake(Function):
    @classmethod
    def forward(cls, ctx, inp1, inp2, training):
        assert inp1.size() == inp2.size()
        gate_size = [inp1.size()[0], *itertools.repeat(1, inp1.dim() - 1)]
        gate = inp1.new(*gate_size)
        if training:
            gate.uniform_(0, 1)
        else:
            gate.fill_(0.5)
        return inp1 * gate + inp2 * (1. - gate)

    @classmethod
    def backward(cls, ctx, grad_output):
        grad_inp1 = grad_inp2 = grad_training = None
        gate_size = [grad_output.size()[0], *itertools.repeat(1,
                                                              grad_output.dim() - 1)]
        gate = Variable(grad_output.data.new(*gate_size).uniform_(0, 1))
        if ctx.needs_input_grad[0]:
            grad_inp1 = grad_output * gate
        if ctx.needs_input_grad[1]:
            grad_inp2 = grad_output * (1 - gate)
        assert not ctx.needs_input_grad[2]
        return grad_inp1, grad_inp2, grad_training


def shake(inp1, inp2, training=False):
    return Shake.apply(inp1, inp2, training)

# todo
class ShiftConvDownsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels=2 * in_channels,
                              out_channels=out_channels,
                              kernel_size=1,
                              groups=2)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = torch.cat((x[:, :, 0::2, 0::2],
                       x[:, :, 1::2, 1::2]), dim=1)
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        return x



class ResNet32x32(nn.Module):
    def __init__(self, args ,block, layers, channels, groups=1, num_classes=100, downsample='basic'):
        super().__init__()
        assert len(layers) == 3
        self.agrs = args

        self.gn = GassianNoise(args, shape=(args.batch_size, 3, 32, 32), std=0.15)

        self.downsample_mode = downsample
        self.inplanes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1,
                               padding=1, bias=False)

        self.layer1 = self._make_layer(block, channels, groups, layers[0])
        self.layer2 = self._make_layer(
            block, channels * 2, groups, layers[1], stride=2)
        self.layer3 = self._make_layer(
            block, channels * 4, groups, layers[2], stride=2)

        self.avgpool = nn.AvgPool2d(8)

        self.fc1 = nn.Linear(block.out_channels(
            channels * 4, groups), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, groups, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != block.out_channels(planes, groups):

            if self.downsample_mode == 'basic' or stride == 1:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, block.out_channels(planes, groups),
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(block.out_channels(planes, groups)),
                )

            elif self.downsample_mode == 'shift_conv':
                downsample = ShiftConvDownsample(in_channels=self.inplanes,
                                                 out_channels=block.out_channels(planes, groups))

            else:
                assert False

        layers = []
        layers.append(block(self.inplanes, planes, groups, stride, downsample))

        self.inplanes = block.out_channels(planes, groups)

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc1(x)





