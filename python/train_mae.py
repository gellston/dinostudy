import torch
import copy
import torch
import torch.nn as nn
import time
import math
import os

import torch.nn.functional as F

from torch.optim.adamw import AdamW

from torch.utils.data import DataLoader
from dataset.dinocropdataset import DinoCropDataset

from model.convnextv2 import convnextv2_atto
from model.projection import projection

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## Hyper Parameter
epochs = 100
lr=1e-4
min_lr=1e-7
weight_decay=1e-5
save_dir = r'C:\github\dinostudy\weights'
## Hyper Parameter



encoder_backbone = convnextv2_atto(in_channels=1).to(device)
