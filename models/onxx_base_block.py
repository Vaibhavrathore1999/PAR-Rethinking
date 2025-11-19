import math
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from einops import rearrange, reduce
from timm.models.layers import trunc_normal_, DropPath


# ---------------------------------------------------------------------------
# IMPORTANT CHANGE:
#  - All mixing / augmentation (trans_data) is removed from forward.
#  - Forward must be deterministic for ONNX graph export.
#  - Put data mixing in the training loop instead.
# ---------------------------------------------------------------------------


def initialize_weights(module):
    for m in module.children():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, _BatchNorm):
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.uniform_(-stdv, stdv)


# -------------------------
# Classifier (ONNX-safe)
# -------------------------
class Classifier(nn.Module):

    def __init__(self, c_in, nattr, bb):
        super(Classifier, self).__init__()
        self.bb = bb
        self.nattr = nattr
        self.c = c_in

        if bb == 'resnet50':
            self.separte = nn.Sequential(
                nn.Linear(c_in, nattr * c_in),
                nn.BatchNorm1d(nattr * c_in),
                nn.ReLU()
            )
        else:
            self.separte = nn.Sequential(
                nn.Linear(c_in, nattr * c_in),
                nn.GELU()
            )

        self.logits = nn.Sequential(
            nn.Linear(c_in, nattr)
        )

    # -------------------------------------------------------------
    # FINAL ONNX-SAFE FORWARD
    #   - no label input
    #   - no "mode"
    #   - no randomness
    #   - deterministic static operations only
    # -------------------------------------------------------------
    def forward(self, x):

        # Reshape only if resnet50 encoded HW features
        if self.bb == 'resnet50':
            # (N, C, H, W) → (N, HW, C) → mean over HW
            x = rearrange(x, 'n c h w -> n (h w) c')
            x = reduce(x, 'n k c -> n c', reduction='mean')

        # Separate -> (N, nattr*c)
        x = self.separte(x)

        # (N, nattr*c) → (N, nattr, c)
        x = x.view(x.shape[0], self.nattr, self.c)

        # Sum attribute features → (N, c)
        x = x.sum(1)

        # Compute classifier logits → (N, nattr)
        logits = self.logits(x)

        return logits


# -------------------------
# Network wrapper
# -------------------------
class Network(nn.Module):
    def __init__(self, backbone, classifier):
        super(Network, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    # ONNX-safe forward:
    #   - **NO labels**
    #   - **NO mode**
    #   - returns list for YOUR original code, but only logits
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return [logits]
