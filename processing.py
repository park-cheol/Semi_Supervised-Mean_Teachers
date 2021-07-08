"""Imagenet Label"""
import numpy as np

def Imagenet(labels_dir):
    with open(labels_dir) as f:
        read = f.read()
        lines = read.splitlines()
        name = [c.split('_') for c in lines]

        label=[]
        for labels in name:
            c = labels[0]
            label.append(c)

        data = {}
        i = 0
        for l in lines:
            data[l] = label[i]
            i += 1

    return data
import torchvision

# with open('data-local/labels/cifar10/10percent (another copy).txt') as f:
#     a= dict(line.split('_') for line in f.read().splitlines())
#     print(a)