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
        c = input.shape[0] # !todo channel? batch?
        self.noise2.data.data.normal_(0, std=self.std1)

        return input + self.noise2[:c] # todo print shape
################
# model
################

class Net(nn.Module):

    def __init__(self, args, std = 0.15):
        super(Net, self).__init__()
        self.args = args
        self.std = std

        self.gn = GassianNoise(shape=(args.batch_size, 3, 32, 32), std=self.std)

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

            self.BNdense = nn.BatchNorm2d(10)

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

        self.dense = nn.Linear(128, 10)

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
            pool_3 = self.pool3(layer_3c)

            reshaped = pool_3.view(-1, 128)
            dense = self.BNdense(self.dense(reshaped))

        else:
            layer_1a = F.leaky_relu(self.conv1a(input), negative_slope=0.1)  # self.BN1a
            layer_1b = F.leaky_relu(self.conv1b(layer_1a), negative_slope=0.1)  # self.BN1b
            layer_1c = F.leaky_relu(self.conv1c(layer_1b), negative_slope=0.1)  # self.BN1c
            drop_pool_1 = self.drop1(self.pool1(layer_1c))

            layer_2a = F.leaky_relu(self.conv2a(drop_pool_1), negative_slope=0.1)  # self.BN2a
            layer_2b = F.leaky_relu(self.conv2b(layer_2a), negative_slope=0.1)  # self.BN2b
            layer_2c = F.leaky_relu(self.conv2c(layer_2b), negative_slope=0.1)  # self.BN2c
            drop_pool_2 = self.drop2(self.pool2(layer_2c))

            layer_3a = F.leaky_relu(self.conv3a(drop_pool_2), negative_slope=0.1)  # self.BN3a
            layer_3b = F.leaky_relu(self.conv3b(layer_3a), negative_slope=0.1)  # self.BN3b
            layer_3c = F.leaky_relu(self.conv3c(layer_3b), negative_slope=0.1)  # self.BN3c
            pool_3 = self.pool3(layer_3c)

            reshaped = pool_3.view(-1, 128)
            dense = self.dense(reshaped)

        return dense

# print 한번찍어보기















