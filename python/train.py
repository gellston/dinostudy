import torch
import copy
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from dataset.dinocropdataset import DinoCropDataset

from model.convnextv2 import convnextv2_atto
from model.projection import projection

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




## Hyper Parameter
epochs = 100
batch_size=1
global_size = 1024
local_size = 448
local_crops_number = 2

## Hyper Parameter


## Head Adapter
out_dim = 65536
hidden_dim = 2048
bottleneck_dim = 256
## Head Adapter





dataset = DinoCropDataset(root_dir=r"C:\github\dataset\dino_test",
                          global_size=1024,
                          local_size=448,
                          global_scale_aug=(0.95, 1.0),
                          local_scale_aug=(0.95, 1.0),
                          local_crops_number=2)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
)


student_backbone = convnextv2_atto().to(device)
teacher_backbone = convnextv2_atto().to(device)

teacher_backbone.load_state_dict(student_backbone.state_dict())

# teacher는 gradient 안 씀
for p in teacher_backbone.parameters():
    p.requires_grad = False

teacher_backbone.eval()



student_proj = projection(embed_dim=320,
                          hidden_dim=hidden_dim,
                          bottleneck_dim=bottleneck_dim)

teacher_proj = projection(embed_dim=320,
                          hidden_dim=hidden_dim,
                          bottleneck_dim=bottleneck_dim)

teacher_proj.load_state_dict(student_proj.state_dict())

for p in teacher_proj.parameters():
    p.requires_grad = False

student_last = torch.nn.utils.parametrizations.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False)).to(device)
teacher_last = torch.nn.utils.parametrizations.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False)).to(device)

teacher_last.load_state_dict(student_last.state_dict())


student_last.parametrizations.weight.original0.data.fill_(1.0)
teacher_last.parametrizations.weight.original0.data.fill_(1.0)

student_last.parametrizations.weight.original0.requires_grad = False
teacher_last.parametrizations.weight.original0.requires_grad = False

for p in teacher_last.parameters():
    p.requires_grad = False

teacher_proj.eval()
teacher_last.eval()

print('test')
