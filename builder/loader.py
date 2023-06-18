# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# modify from
# https://github.com/facebookresearch/moco-v3/blob/main/moco/loader.py

from PIL import Image, ImageFilter, ImageOps
import random

DATASETS_STATS = {"cifar10": [(0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)],
         "cifar100": [(0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)],
        "imagenet": [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]
                  }

class SingleCropsTransform:
    """Take a single random crop of one image"""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return self.base_transform(x)

class DoubleCropsTransform:
    """Take two random crops of one image"""

    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x):
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)
        return [im1, im2]


class MultiCropsTransform:
    """Take multiple random crops of one image"""

    def __init__(self, base_transform1, base_transform2, small_transform, snum):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2
        self.small_transform = small_transform
        self.snum = snum

    def __call__(self, x):
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)
        simgs = []
        for i in range(0, self.snum):
            simgs.append(self.small_transform(x))
        return [im1, im2, simgs]


class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __call__(self, x):
        return ImageOps.solarize(x)