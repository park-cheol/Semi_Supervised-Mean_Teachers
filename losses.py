import torch
import torch.nn.functional as F

# target에는 send gradient X
def softmax_mes_loss(input_logits, target_logits):
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1] # channel

    return F.mse_loss(input_softmax, target_softmax, size_average=False) / num_classes
# size_average(default:None) = 마지막에 1/N을 관한 것 False로 안해주면 taget과 input
# 의 값들이 다 더해져서 나눠짐 즉 2 * num_classes로 나눠짐

# mse_loss가 더 좋다고 함
def softmax_kl_loss(input_logits, target_logits):
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    return F.kl_div(input_log_softmax, target_softmax, size_average=False)

# F.mse loss와 비슷하지만 양방향으로 gradients를 보냄
def symmetric_mse_loss(input1, input2):
    assert input1.size() == input2.size()
    num_classes = input1.size()[1]

    return torch.sum((input1 - input2) ** 2) / num_classes

