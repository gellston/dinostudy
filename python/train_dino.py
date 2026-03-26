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
batch_size=3
global_size = 1024
local_size = 224
local_crops_number = 6
lr=1e-4
min_lr=1e-7
weight_decay=1e-5
save_dir = r'C:\github\dinostudy\weights'
## Hyper Parameter


## Special Hyper Param
embed_dim = 320
out_dim = 65536
hidden_dim = 2048
bottleneck_dim = 256

student_temp = 0.1
teacher_temp_warmup = 0.04
teacher_temp_final = 0.07
teacher_temp_warmup_epochs = 10


#center_momentum = 0.9
center_momentum = 0.999
teacher_momentum_base = 0.996
## Special Hyper Param




use_amp = torch.cuda.is_available()
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)



dataset = DinoCropDataset(root_dir=r"C:\github\dataset\dino_test",
                          global_size=global_size,
                          local_size=local_size,
                          global_scale_aug=(0.95, 1.05),
                          local_scale_aug=(0.95, 1.05),
                          local_crops_number=local_crops_number)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
)


student_backbone = convnextv2_atto(in_channels=1).to(device)
teacher_backbone = convnextv2_atto(in_channels=1).to(device)

teacher_backbone.load_state_dict(student_backbone.state_dict())

# teacher는 gradient 안 씀
for p in teacher_backbone.parameters():
    p.requires_grad = False

teacher_backbone.eval()



student_proj = projection(embed_dim=embed_dim,
                          hidden_dim=hidden_dim,
                          bottleneck_dim=bottleneck_dim).to(device)

teacher_proj = projection(embed_dim=embed_dim,
                          hidden_dim=hidden_dim,
                          bottleneck_dim=bottleneck_dim).to(device)

teacher_proj.load_state_dict(student_proj.state_dict())

for p in teacher_proj.parameters():
    p.requires_grad = False


# 2. Last Layer 설정
student_last = torch.nn.utils.parametrizations.weight_norm(
    nn.Linear(bottleneck_dim, out_dim, bias=False)
)
teacher_last = torch.nn.utils.parametrizations.weight_norm(
    nn.Linear(bottleneck_dim, out_dim, bias=False)
)

# [중요] 초기화: 1.0으로 채우지 말고 랜덤하게 두세요. 
# weight_norm은 내부적으로 original0(g)과 original1(v)로 분리됩니다.

# 3. 가중치 복사
teacher_last.load_state_dict(student_last.state_dict())

# 4. 크기(Magnitude)만 고정하고 방향(Direction)은 학습 허용
# PyTorch parametrizations에서 original0은 보통 g(scale)입니다.
student_last.parametrizations.weight.original0.requires_grad = False 
teacher_last.parametrizations.weight.original0.requires_grad = False

for p in teacher_last.parameters():
    p.requires_grad = False

student_last.to(device)
teacher_last.to(device)


optimizer = AdamW(
    list(student_backbone.parameters()) +
    list(student_proj.parameters()) +
    list(student_last.parameters()),
    lr=lr,
    weight_decay=weight_decay,
)

center = torch.zeros(1, out_dim, device=device)

total_steps = epochs * len(loader)
global_step = 0

print("dataset size =", len(dataset))
print("steps per epoch =", len(loader))
print("total steps =", total_steps)
print("device =", device)


best_loss = 999999
for epoch in range(epochs):
    student_backbone.train()
    student_proj.train()
    student_last.train()

    teacher_backbone.eval()
    teacher_proj.eval()
    teacher_last.eval()


    # teacher temperature warmup
    if epoch < teacher_temp_warmup_epochs:
        teacher_temp = teacher_temp_warmup + (teacher_temp_final - teacher_temp_warmup) * (epoch / max(1, teacher_temp_warmup_epochs - 1))
    else:
        teacher_temp = teacher_temp_final

    epoch_loss = 0.0
    epoch_start = time.time()



    for it, batch in enumerate(loader):
        # ---------------------------------------------
        # cosine lr schedule
        # ---------------------------------------------
        lr_now = min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * global_step / total_steps))
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        # teacher EMA momentum schedule
        m = 1.0 - (1.0 - teacher_momentum_base) * (math.cos(math.pi * global_step / total_steps) + 1.0) * 0.5

        global_crops = list(batch["global_crops"])
        local_crops = list(batch["local_crops"])
        all_crops = global_crops + local_crops

        # ---------------------------------------------
        # student forward: global + local 모두 사용
        # ---------------------------------------------
        student_out = []

        with torch.cuda.amp.autocast(enabled=use_amp):
            for crop in all_crops:
                crop = crop.to(device, non_blocking=True).float()

                # 0~1 -> 정규화
                crop = (crop - 0.5) / 0.5

                feat = student_backbone.forward_features(crop)

                if isinstance(feat, (tuple, list)):
                    feat = feat[-1]

                if isinstance(feat, dict):
                    if "feat" in feat:
                        feat = feat["feat"]
                    elif "features" in feat:
                        feat = feat["features"]
                    else:
                        feat = list(feat.values())[-1]

                if feat.ndim == 4:
                    feat = feat.mean(dim=(2, 3))

                feat = student_proj(feat)
                feat = F.normalize(feat, dim=-1)
                logits = student_last(feat)
                student_out.append(logits)

            # -----------------------------------------
            # teacher forward: global crop 2개만 사용
            # -----------------------------------------
            with torch.no_grad():
                teacher_out = []

                for crop in global_crops:
                    crop = crop.to(device, non_blocking=True).float()
                    crop = (crop - 0.5) / 0.5

                    feat = teacher_backbone.forward_features(crop)

                    if isinstance(feat, (tuple, list)):
                        feat = feat[-1]

                    if isinstance(feat, dict):
                        if "feat" in feat:
                            feat = feat["feat"]
                        elif "features" in feat:
                            feat = feat["features"]
                        else:
                            feat = list(feat.values())[-1]

                    if feat.ndim == 4:
                        feat = feat.mean(dim=(2, 3))

                    feat = teacher_proj(feat)
                    feat = F.normalize(feat, dim=-1)
                    logits = teacher_last(feat)
                    teacher_out.append(logits)

            # -----------------------------------------
            # DINO loss
            # teacher는 center + temp 적용 후 softmax
            # student는 temp 적용 후 log_softmax
            # -----------------------------------------
            teacher_probs = []
            for t_out in teacher_out:
                t_prob = F.softmax((t_out - center) / teacher_temp, dim=-1).detach()
                teacher_probs.append(t_prob)

            student_log_probs = []
            for s_out in student_out:
                s_log_prob = F.log_softmax(s_out / student_temp, dim=-1)
                student_log_probs.append(s_log_prob)

            loss = 0.0
            n_terms = 0

            # teacher view 2개(global) vs student all views(global+local)
            # 같은 index의 global-global 짝은 제외
            for iq, q in enumerate(teacher_probs):
                for v, s in enumerate(student_log_probs):
                    if v == iq:
                        continue
                    loss_term = torch.sum(-q * s, dim=-1).mean()
                    loss = loss + loss_term
                    n_terms += 1

            loss = loss / n_terms

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # ---------------------------------------------
        # teacher EMA update
        # ---------------------------------------------
        with torch.no_grad():
            for ps, pt in zip(student_backbone.parameters(), teacher_backbone.parameters()):
                pt.data.mul_(m).add_((1.0 - m) * ps.data)

            for ps, pt in zip(student_proj.parameters(), teacher_proj.parameters()):
                pt.data.mul_(m).add_((1.0 - m) * ps.data)

            for ps, pt in zip(student_last.parameters(), teacher_last.parameters()):
                pt.data.mul_(m).add_((1.0 - m) * ps.data)

            # center update
            batch_center = torch.cat(teacher_out, dim=0).mean(dim=0, keepdim=True)
            center = center * center_momentum + batch_center * (1.0 - center_momentum)

        epoch_loss += loss.item()
        global_step += 1

        if (it + 1) % 20 == 0 or (it + 1) == len(loader):
            print(
                f"[Epoch {epoch+1:03d}/{epochs:03d}] "
                f"[Iter {it+1:04d}/{len(loader):04d}] "
                f"loss={loss.item():.4f} "
                f"lr={lr_now:.6e} "
                f"teacher_temp={teacher_temp:.4f} "
                f"m={m:.6f}"
            )

    avg_loss = epoch_loss / len(loader)
    epoch_time = time.time() - epoch_start

    print(
        f"==> Epoch {epoch+1:03d} done | "
        f"avg_loss={avg_loss:.4f} | "
        f"time={epoch_time:.1f}s"
    )



    # 1. Best Model 저장 (로스 기준)
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            'epoch': epoch + 1,
            'student_backbone': student_backbone.state_dict(),
            'student_proj': student_proj.state_dict(),
            'student_last': student_last.state_dict(),
            'teacher_backbone': teacher_backbone.state_dict(),
            'optimizer': optimizer.state_dict(),
            'center': center,
            'loss': avg_loss,
        }, os.path.join(save_dir, "best_dino_model.pth"))
        print(f"--- Best model saved with loss: {best_loss:.4f} ---")

    # 2. 주기적 저장 (예: 10 에포크마다)
    if (epoch + 1) % 10 == 0:
        torch.save(student_backbone.state_dict(), 
                   os.path.join(save_dir, f"student_backbone_ep{epoch+1}.pth"))
