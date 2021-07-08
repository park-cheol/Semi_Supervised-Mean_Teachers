import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd.variable import Variable

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

        self.gn = GassianNoise(args, shape=(args.batch_size, 3, 128, 128), std=self.std)

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
            nn.Linear(4096, 1000)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels # w * d * h 부피
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, input):
        conv1_a = F.relu(self.bn1_a(self.conv1_a(input)))
        conv1_b = F.relu(self.bn1_b(self.conv1_b(conv1_a)))
        maxpool1 = self.maxpool1(conv1_b)

        conv2_a = F.relu((self.bn2_a(self.conv2_a(maxpool1))))
        conv2_b = F.relu((self.bn2_b(self.conv2_b(conv2_a))))
        maxpool2 = self.maxpool2(conv2_b)

        conv3_a = F.relu((self.bn3_a(self.conv3_a(maxpool2))))
        conv3_b = F.relu((self.bn3_b(self.conv3_b(conv3_a))))
        conv3_c = F.relu((self.bn3_c(self.conv3_c(conv3_b))))
        maxpool3 = self.maxpool3(conv3_c)

        conv4_a = F.relu((self.bn4_a(self.conv4_a(maxpool3))))
        conv4_b = F.relu((self.bn4_b(self.conv4_b(conv4_a))))
        conv4_c = F.relu((self.bn4_c(self.conv4_c(conv4_b))))
        maxpool4 = self.maxpool4(conv4_c)

        conv5_a = F.relu((self.bn5_a(self.conv5_a(maxpool4))))
        conv5_b = F.relu((self.bn5_b(self.conv5_b(conv5_a))))
        conv5_c = F.relu((self.bn5_c(self.conv5_c(conv5_b))))
        maxpool5 = self.maxpool5(conv5_c)

        avgpool = self.avgpool(maxpool5)

        reshaped = torch.flatten(avgpool, 1)
        predict = self.classifier(reshaped)

        return predict








###################################
# RESNET MODEL
###################################
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, args, block, num_blocks, num_classes=10, std = 0.15):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.std = std
        self.args = args

        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.5)

        self.gn = GassianNoise(args, shape=(args.batch_size, 3, 32, 32), std=self.std)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.training:  # todo self.training을 지정?
            x = self.gn(x)
        conv_bn1_relu = F.relu(self.bn1(self.conv1(x)))
        layer1 = self.layer1(conv_bn1_relu)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        avg_pool2d = F.avg_pool2d(layer4, 4)
        reshaped = avg_pool2d.view(avg_pool2d.size(0), -1)
        out = self.linear(reshaped)
        return out








###################################
# RESNET Model (무기한 폐쇠)
###################################
""" 작동이 정상적이지 못함 X"""
# class BottleneckBlock(nn.Module):
#     @classmethod
#     def out_channels(cls, channels, groups): # todo out_channels?
#         if groups > 1:
#             return 2 * channels
#         else:
#             return 4 * channels
#
#     def __init__(self, in_channels, out_channels, groups, stride=1, downsample=None):
#         super(BottleneckBlock, self).__init__()
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#
#         self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
#         self.bn_1 = nn.BatchNorm2d(out_channels)
#
#         self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, groups=groups)
#         self.bn_2 = nn.BatchNorm2d(out_channels)
#
#         self.conv_3 = nn.Conv2d(out_channels, self.out_channels(out_channels, groups), kernel_size=1, bias=False)
#         self.bn_3 = nn.BatchNorm2d(self.out_channels(out_channels, groups))
#
#     def forward(self, input):
#         conv_1 = self.conv_1(input)
#         bn_1 = self.bn_1(conv_1)
#         relu_1 = self.relu(bn_1)
#
#         conv_2 = self.conv_2(relu_1)
#         bn_2 = self.bn_2(conv_2)
#         relu_2 = self.relu(bn_2)
#
#         conv_3 = self.conv_3(relu_2)
#         bn_3 = self.bn_3(conv_3)
#
#         if self.downsample is not None:
#             input = self.downsample(input)
#
#         return self.relu(input + bn_3)
#
# # LAYERS[4, 4, 4] , channels=96, downsample='basic', **kwargs)
# class ResNet(nn.Module):
#
#     def __init__(self, block, layers, channels, groups=1, num_classes=10, downsample='basic'):
#         super(ResNet, self).__init__()
#         assert len(layers) == 3
#         self.downsample_mode = downsample
#         self.in_channels = 16
#
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
#
#         self.layer1 = self._make_layer(block, channels, groups, layers[0])
#         self.layer2 = self._make_layer(block, channels * 2, groups, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, channels * 4, groups, layers[2], stride=2)
#
#         self.avgpool = nn.AvgPool2d(8) # todo out_channels 어떻게 찍히는가
#         self.fc1 = nn.Linear(block.out_channels(channels * 4, groups), num_classes)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def _make_layer(self, block, out_channels, groups, blocks, stride=1):
#         if stride != 1 or self.in_channels != block.out_channels(out_channels, groups):
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.in_channels, block.out_channels(out_channels, groups), kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(block.out_channels(out_channels, groups))
#             )
#
#         else:
#             assert False
#
#         layers = []
#         layers.append(block(self.in_channels, out_channels, groups, stride, downsample))
#         self.in_channels = block.out_channels(out_channels, groups)
#
#         for i in range(1, blocks):
#             layers.append(block(self.in_channels, out_channels, groups))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, input):
#         conv1 = self.conv1(input)
#         layer1 = self.layer1(conv1)
#         layer2 = self.layer2(layer1)
#         layer3 = self.layer3(layer2)
#         avgpool = self.avgpool(layer3)
#         reshaped = avgpool.view(input.size(0), -1)
#         output = self.fc1(reshaped)
#
#         return output

















