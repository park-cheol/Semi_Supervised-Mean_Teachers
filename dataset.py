import itertools
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data.sampler import Sampler

NO_LABEL = -1

class RandomTranslateWithReflect:
    """이미지를 랜덤으로 이동
    [-max, max]사이에서 독립적으로 uniformly하게 샘플링된 정수를 수직, 수평 방향으로 n pixel만큼 이동
    그 blank 된 공간은 reflect padding으로 채워짐
    """

    def __init__(self, max_translation):
        self.max_translation = max_translation

    # 클래스는 함수의 parameter로도 사용 가능함 그러므로 클래스를 함수로써 호출 가능하게 하기 위하여 만든 메서드
    def __call__(self, old_image):
        xtranslation, ytranslation = np.random.randint(-self.max_translation,
                                                       self.max_translation + 1,
                                                       size=2)
        xpad, ypad = abs(xtranslation), abs(ytranslation)
        xsize, ysize = old_image.size # 32, 32


        flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT) # 좌 우
        flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM) # 위아래
        flipped_both = old_image.transpose(Image.ROTATE_180) # 둘다 plt로 확인해봄

        # 옆, 위아래니까 * 2
        new_image = Image.new("RGB", (xsize + 2 * xpad, ysize + 2 * ypad)) # 현재는 그저 검은 화면
        # Image.new(mode, size): 주어진 형식의 새로운 이미지를 생성

        new_image.paste(old_image, (xpad, ypad)) # xpad ,ypad 랜덤으로 원본그림 크기 복사

        new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_image.paste(flipped_lr, (xpad - xsize + 1, ypad)) # 위 아래만 검은색

        new_image.paste(flipped_tb, (xpad, ypad + ysize - 1)) # 위 전부 아래 양옆모서리 검은색
        new_image.paste(flipped_tb, (xpad, ypad - ysize + 1)) # 위 아래 양 옆모서리 검은색

        new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

        new_image = new_image.crop((xpad - xtranslation,
                                    ypad - ytranslation,
                                    xpad + xsize - xtranslation,
                                    ypad + ysize - ytranslation))

        return new_image

# student와 teacher에게 같은 분포에서 각각 독립적으로 noise sampling함
# 그걸 구현하기위해 transformTwice 클래스로 컨트롤함
# 하나는 input / 다른 것 ema_input
class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, input):
        out1 = self.transform(input)
        out2 = self.transform(input)

        return out1, out2



# unlabeled data와 labeled data를 나누기
# dataset: Imagefolder로 불러온 image들
# labels: dict으로 되어있는 것 {"image idx": "label"} ex) {"16303.jpg": airplane}
# 즉 labels => labeling 되어있는 데이터
def relabel_dataset(dataset, labels):
    unlabeled_idxs = []
    for idx in range(len(dataset.imgs)):
        path, _ = dataset.imgs[idx] # dataset은 torchvision Imagefolder인데 .imgs따로 있음 이미지 불러오기
        filename = os.path.basename(path)
        # os.path.abspath(path)와 상반: abspath는 절대경로로 home/jun2/....식으로 반환
        # os.path.basename(path): 입력받은 경로의 기본 이름을 반환 즉 "path"만

        if filename in labels: # labeling되어 있는 데이터 중에 filename data 여부
            label_idx = dataset.class_to_idx[labels[filename]]
            # class_to_idx : 클래스의 인덱스 반환 e.g. {"airplane": 0...} (.classes 하면 class를 반환)
            # labeled data의 인덱스 가져오기
            dataset.imgs[idx] = path, label_idx
            del labels[filename]

        else:
            dataset.imgs[idx] = path, NO_LABEL
            unlabeled_idxs.append(idx)

    # for 문을 다 돌고 labels의 dict에 아무것도 없어야함 있을 시 에러 처리
    if len(labels) != 0:
        message = "List of unlabeled contains {} unknown files: {}, ..."
        some_missing = ', '.join(list(labels.keys())[:5])
        raise LookupError(message.format(len(labels), some_missing))

    labeled_idxs = sorted(set(range(len(dataset.imgs))) - set(unlabeled_idxs))
    # set: 순서가 없고 unique한 값을 가짐 즉 key가 없고 value만 존재
    # 즉 집합-집합으로 총 dataset 이미지 중에서 unlabeled 된 것을 제외
    return labeled_idxs, unlabeled_idxs # 둘 다 type: list

# 총 batch(ex.256)에서 labeled data(64)를 무적권 넣는 것
# 즉 불러올 데이터에 labeled data를 넣기 위한 작업
class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    # primary: unlabeled / secondary: labeled
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.primary_batch_size = batch_size - secondary_batch_size
        self.secondary_batch_size = secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0


    def __iter__(self):

        primary_iter = iterate_once(self.primary_indices) # unlabeled iteration 마다 한번
        secondary_iter = iterate_eternally(self.secondary_indices) # 계속

        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)
# permutation(순열): 순서가 부여된 집합을 임의의 다른 순서로 뒤섞는 연산
# np.random.shuffle은 inplac임
# 만약 permutation(5) 처럼 정수를 받을 경우 np.arange(5)를 한 후에 자동으로 셔플한 값을 return

"""계속 순서를 섞음 indices범위안에서"""
def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())

# Return VS yield
# itrables: 객체를 순환하며 하나씩 꺼내서 사용함 이는 큰 메모리에 대해서 별로 좋지 않음
# generator: 모든 값을 메모리에 담고 있지 않고 그때 그때 값을 생성하고 반환한 후 아예 잊음 하나씩 처리해가며 순환
# return은 main에서 함수 호출하면 함수에서 실행되고 다시 main으로 돌아와서 나머지 실행
# yiled은 함수에서 yiled된 값을 반환 후 다시 함수로 또 돌아옴 조건이 만족되는 한

# chain.from_iterable(): 1차원 list로 만들어주는 것

# todo 이걸 사용하는 목적
def grouper(iterable, n):
    "고정된 길이의 block으로 data를 수집"
    # grouper('ABCDEF', 3) --->('A','B','C'), ('D,'E','F')
    args = [iter(iterable)] * n
    return zip(*args)




















