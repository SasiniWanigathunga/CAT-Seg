# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union
from einops import rearrange

import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..transformer.cat_seg_predictor import CATSegPredictor


@SEM_SEG_HEADS_REGISTRY.register()
class CATSegHead(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        num_classes: int,
        ignore_value: int = -1,
        # extra parameters
        feature_resolution: list,
        transformer_predictor: nn.Module,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            num_classes: number of classes to predict
            ignore_value: category id to be ignored during training.
            feature_resolution: resolution of the feature map
            transformer_predictor: the transformer decoder that makes prediction
        """
        super().__init__()
        self.ignore_value = ignore_value
        self.predictor = transformer_predictor
        self.num_classes = num_classes
        self.feature_resolution = feature_resolution

        with open('print.txt', 'a') as f:
            print('feature_resolution:', feature_resolution, file=f)
            print('transformer_predictor:', transformer_predictor, file=f)
            print('num_classes:', num_classes, file=f)
            print('ignore_value:', ignore_value, file=f)

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        with open('print.txt', 'a') as f:
            print('cfg:', cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON, file=f)
        return {
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "feature_resolution": cfg.MODEL.SEM_SEG_HEAD.FEATURE_RESOLUTION,
            "transformer_predictor": CATSegPredictor(
                cfg,
            ),
        }

    def forward(self, features, guidance_features, prompt=None, gt_cls=None):
        """
        Arguments:
            img_feats: (B, C, HW)
            guidance_features: (B, C, )
        """
        img_feat = rearrange(features[:, 1:, :], "b (h w) c->b c h w", h=self.feature_resolution[0], w=self.feature_resolution[1])
        return self.predictor(img_feat, guidance_features, prompt, gt_cls)