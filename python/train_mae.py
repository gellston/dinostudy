import torch
import copy
import torch
import torch.nn as nn
import time
import math
import os



from utils.sparse import make_cur_active

from utils.helper import copy_weights_ignore_name

from model.convnextv2 import convnextv2_atto
from model.convnextv2_mae import convnextv2_mae_atto
from model.projection import projection


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")






## Hyper Parameter
epochs = 100
lr=1e-4
min_lr=1e-7
weight_decay=1e-5
save_dir = r'C:\github\dinostudy\weights'
## Hyper Parameter



encoder_backbone_normal = convnextv2_atto(in_channels=1).to(device)
encoder_backbone2_mae = convnextv2_mae_atto(in_channels=1).to(device)


copy_weights_ignore_name(encoder_backbone2_mae, encoder_backbone_normal)

x = torch.randn(2, 1, 224, 224).to(device)


make_cur_active(2, 56, 56, 1.0, device=x.device)


y1 = encoder_backbone_normal(x)
y2 = encoder_backbone2_mae(x)

print('test')