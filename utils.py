import sys
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable

##########################
# Ramps up/ Ramps Down
##########################

# rampup_length [0, 1]사이의 값
# consistency loss와 learning rate에 사용
# We applied a ramp-up period of 40000 training steps at the beginning of training
def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        # np.clip(array, min, max): array를 min max안으로 좁힘
        phase = 1.0 - current / rampup_length

        return float(np.exp(-5.0 * phase * phase))

def linear_rampup(current, rampup_length):
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length

# todo cosine annealing paper 읽기
# decay learning rate할때 사용 (resnet 구조)
def cosine_rampdown(current, rampdown_length):
    assert 0 <= current <= rampdown_length

    # 0.5[cos(pi * x) + 1]
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))

"""cosine annealing 간단설명
학습율의 max, min value을 정해서 그 범위의 학습율에 cosine function을 이용하여 scheduling
장점: cosine을 이용하여 급격히 증가시켰다가 급격히 감소시키기 때문에 모델의 manifold 공간의 안장에 빠르게
벗어 날 수 있고, 또한 학습 중간에 생기는 정체 구간들을 빠르게 벗어날 수 있다.
"""



###############################
# todo 좀더 조사
###############################
def parameters_string(module):
    lines = [
        "",
        "List of model parameters:",
        "=========================",
    ]

    row_format = "{name:<40} {shape:>20} ={total_size:>12,d}"
    params = list(module.named_parameters())
    for name, param in params:
        lines.append(row_format.format(
            name=name,
            shape=" * ".join(str(p) for p in param.size()),
            total_size=param.numel()
        ))
    lines.append("=" * 75)
    lines.append(row_format.format(
        name="all parameters",
        shape="sum of above",
        total_size=sum(int(param.numel()) for name, param in params)
    ))
    lines.append("")
    return "\n".join(lines)


def assert_exactly_one(lst):
    assert sum(int(bool(el)) for el in lst) == 1, ", ".join(str(el)
                                                            for el in lst)


class AverageMeterSet:
    def __init__(self):
        self.meters = {}

    def __getitem__(self, key):
        return self.meters[key]

    def update(self, name, value, n=1):
        if not name in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, postfix=''):
        return {name + postfix: meter.val for name, meter in self.meters.items()}

    def averages(self, postfix='/avg'):
        return {name + postfix: meter.avg for name, meter in self.meters.items()}

    def sums(self, postfix='/sum'):
        return {name + postfix: meter.sum for name, meter in self.meters.items()}

    def counts(self, postfix='/count'):
        return {name + postfix: meter.count for name, meter in self.meters.items()}


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
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

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)


def export(fn):
    mod = sys.modules[fn.__module__]
    if hasattr(mod, '__all__'):
        mod.__all__.append(fn.__name__)
    else:
        mod.__all__ = [fn.__name__]
    return fn


def parameter_count(module):
    return sum(int(param.numel()) for param in module.parameters())



















