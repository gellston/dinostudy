import os
import glob
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class DinoCropDataset(Dataset):
    def __init__(
        self,
        root_dir,
        global_size=224,
        local_size=96,
        global_scale_aug=(0.95, 1.0),
        local_scale_aug=(0.95, 1.0),
        local_crops_number=6,
    ):
        self.root_dir = root_dir
        self.global_size = global_size
        self.local_size = local_size
        self.global_scale_aug = global_scale_aug
        self.local_scale_aug = local_scale_aug
        self.local_crops_number = local_crops_number

        self.image_paths = []
        exts = ["*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff",
                "*.JPG", "*.JPEG", "*.BMP", "*.TIF", "*.TIFF"]

        for ext in exts:
            self.image_paths.extend(glob.glob(os.path.join(root_dir, ext)))

        self.image_paths = list(set(self.image_paths))
        self.image_paths = sorted(self.image_paths)

        if len(self.image_paths) == 0:
            raise ValueError(f"이미지를 찾지 못했습니다: {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"이미지 로드 실패: {path}")

        h, w = img.shape[:2]
        global_crops = []
        g_coords = [] # 글로벌 뷰 2개의 좌표 저장 [(y1, x1, h, w), ...]

        # --- 1. Global View 2개 생성 ---
        for _ in range(2):
            scale = random.uniform(self.global_scale_aug[0], self.global_scale_aug[1])
            crop_h = min(h, max(1, int(round(self.global_size * scale))))
            crop_w = min(w, max(1, int(round(self.global_size * scale))))
            
            y1 = random.randint(0, h - crop_h) if h > crop_h else 0
            x1 = random.randint(0, w - crop_w) if w > crop_w else 0
            
            # 좌표 저장 (나중에 로컬 뷰가 이 안에서 놀 수 있도록)
            g_coords.append((y1, x1, crop_h, crop_w))
            
            crop = img[y1:y1 + crop_h, x1:x1 + crop_w]
            if random.random() < 0.5:
                crop = cv2.flip(crop, 1)
            
            crop = cv2.resize(crop, (self.global_size, self.global_size), interpolation=cv2.INTER_CUBIC)
            crop = torch.from_numpy(crop.astype(np.float32) / 255.0).unsqueeze(0)
            global_crops.append(crop)

        # --- 2. Local View 생성 (글로벌 박스 2개 중 랜덤 선택하여 그 안에서 추출) ---
        local_crops = []
        for _ in range(self.local_crops_number):
            # 두 개의 글로벌 박스 중 하나를 랜덤하게 선택 (고르게 샘플링)
            target_g_y1, target_g_x1, target_g_h, target_g_w = random.choice(g_coords)
            
            scale = random.uniform(self.local_scale_aug[0], self.local_scale_aug[1])
            # 선택된 글로벌 박스의 크기 내에서 로컬 크롭 사이즈 결정
            l_crop_h = min(target_g_h, max(1, int(round(self.local_size * scale))))
            l_crop_w = min(target_g_w, max(1, int(round(self.local_size * scale))))
            
            # 선택된 글로벌 박스 내부 좌표에서만 랜덤 추출
            y1 = target_g_y1 + (random.randint(0, target_g_h - l_crop_h) if target_g_h > l_crop_h else 0)
            x1 = target_g_x1 + (random.randint(0, target_g_w - l_crop_w) if target_g_w > l_crop_w else 0)
            
            crop = img[y1:y1 + l_crop_h, x1:x1 + l_crop_w]
            if random.random() < 0.5:
                crop = cv2.flip(crop, 1)
                
            crop = cv2.resize(crop, (self.local_size, self.local_size), interpolation=cv2.INTER_CUBIC)
            crop = torch.from_numpy(crop.astype(np.float32) / 255.0).unsqueeze(0)
            local_crops.append(crop)

        return {
            "global_crops": global_crops,
            "local_crops": local_crops,
            "path": path
        }