import torch
import copy
import torch
import torch.nn as nn
import time
import math
import os
import time



from torch.utils.data import DataLoader
from torch.optim.radam import RAdam



from utils.sparse import make_cur_active
from utils.helper import copy_weights_ignore_name

from model.convnextv2 import convnextv2_atto
from model.convnextv2_mae import convnextv2_mae_atto
from dataset.maedataset import MAEDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



## Hyper Parameter
epochs = 100
lr=1e-4
weight_decay=1e-5
global_size=1024
batch_size=1
save_dir = r'C:\github\dinostudy\weights'
## Hyper Parameter



encoder_backbone_normal = convnextv2_atto(in_channels=1).to(device)
encoder_backbone_mae = convnextv2_mae_atto(in_channels=1).to(device)


dataset = MAEDataset(root_dir=r"C:\github\dataset\dino_test",
                     global_size=global_size,
                     global_scale_aug=(0.95, 1.05))

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
)




optimizer = RAdam(encoder_backbone_mae.parameters(), lr=lr, weight_decay=weight_decay)



total_steps = epochs * len(loader)
global_step = 0

print("dataset size =", len(dataset))
print("steps per epoch =", len(loader))
print("total steps =", total_steps)
print("device =", device)


best_loss = 999999
for epoch in range(epochs):
    encoder_backbone_mae.train()

    epoch_loss = 0.0
    for it, batch in enumerate(loader):
    
        global_crops = list(batch["global_crops"])
        # ---------------------------------------------
        # student forward: global + local 모두 사용
        # ---------------------------------------------
        student_out = []

        

    avg_loss = epoch_loss / len(loader)


    for it, batch in enumerate(loader):
        print('test here and insert source code')



    # 1. Best Model 저장 (로스 기준)
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            'epoch': epoch + 1,
            'encoder_backbone_normal': encoder_backbone_normal.state_dict(),
            'encoder_backbone_mae': encoder_backbone_mae.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': avg_loss,
        }, os.path.join(save_dir, "best_dino_model.pth"))
        print(f"--- Best model saved with loss: {best_loss:.4f} ---")

    # 2. 주기적 저장 (예: 10 에포크마다)
    if (epoch + 1) % 10 == 0:
        torch.save(encoder_backbone_mae.state_dict(), 
                os.path.join(save_dir, f"encoder_backbone_mae{epoch+1}.pth"))






















# copy_weights_ignore_name(encoder_backbone2_mae, encoder_backbone_normal)

# x = torch.randn(1, 1, 512, 512).to(device)


# make_cur_active(1, 128, 128, 1.0, device=x.device)


# y1 = encoder_backbone_normal(x)
# y2 = encoder_backbone2_mae(x)









# # 시간 측정 변수 초기화

# frame_count = 0
# start_time = time.time()
# fps = 0

# while True:
#     # --- 프레임 처리 로직 (예: 이미지 읽기, 추론 등) ---
#     y1 = encoder_backbone_normal(x) # 가상의 처리 시간 (100fps 목표 시)
#     frame_count += 1
#     # ---------------------------------------------
   

#     # 1초 경과 확인
#     current_time = time.time()
#     elapsed_time = current_time - start_time
    
#     if elapsed_time >= 1.0:
#         fps = frame_count / elapsed_time
#         print(f"FPS: {fps:.2f}")
        
#         # 카운터 및 시간 초기화
#         frame_count = 0
#         start_time = current_time

# print('test')

