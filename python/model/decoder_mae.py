import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder_MAE(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_classes=1
    ):
        super().__init__()
        

    def forward(self, x):
        return x
