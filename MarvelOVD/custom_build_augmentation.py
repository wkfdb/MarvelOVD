# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
import pycocotools.mask as mask_util
import torch
from fvcore.common.file_io import PathManager
from PIL import Image


from detectron2.data import transforms as T
from .transforms.custom_augmentation_impl import EfficientDetResizeCrop

def build_custom_augmentation(cfg, is_train, scale=None, size=None, \
    min_size=None, max_size=None):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.

    Returns:
        list[Augmentation]
    """
    if is_train:
        scale = cfg.INPUT.SCALE_RANGE if scale is None else scale
        size = cfg.INPUT.TRAIN_SIZE if size is None else size
    else:
        scale = (1, 1)
        size = cfg.INPUT.TEST_SIZE
    augmentation = [EfficientDetResizeCrop(size, scale)]

    if is_train:
        augmentation.append(T.RandomFlip())
    # return augmentation
    return T.AugmentationList(augmentation)

# build_custom_transform_gen = build_custom_augmentation
"""
Alias for backward-compatibility.
"""