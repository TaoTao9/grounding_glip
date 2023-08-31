from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BatchNorm2d, SyncBatchNorm

from maskrcnn_benchmark.layers import FrozenBatchNorm2d, NaiveSyncBatchNorm2d
from maskrcnn_benchmark.layers import Conv2d, DFConv2d, SELayer
from maskrcnn_benchmark.modeling.make_layers import group_norm
from maskrcnn_benchmark.utils.registry import Registry

class self_Attention_Tansfer(nn.Module):
    def __init__(self, cfg):
        super(self_Attention_Tansfer, self).__init__()

        # If we want to use the cfg in forward(), then we should make a copy
        # of it and store it for later use:
        # self.cfg = cfg.clone()

        # Translate string names to implementations
        

        # Optionally freeze (requires_grad=False) parts of the backbone
        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT)

    def _freeze_backbone(self, freeze_at):
        if freeze_at < 0:
            return
        for stage_index in range(freeze_at):
            if stage_index == 0:
                m = self.stem  # stage 0 is the stem
            else:
                m = getattr(self, "layer" + str(stage_index))
            for p in m.parameters():
                p.requires_grad = False

    def forward(self, x):
        outputs = []
        #we focus here is to implement the corresponding adapter
        return outputs
    
# def build_self_attention_tansfer(cfg):

#     return self_attention_tansfer(cfg)