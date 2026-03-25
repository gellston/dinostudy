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
        exts = [
            "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff",
            "*.JPG", "*.JPEG", "*.BMP", "*.TIF", "*.TIFF"
        ]

        for ext in exts:
            self.image_paths.extend(glob.glob(os.path.join(root_dir, ext)))

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
        crops = []

        for crop_idx in range(2 + self.local_crops_number):
            if crop_idx < 2:
                out_size = self.global_size
                scale_min, scale_max = self.global_scale_aug
            else:
                out_size = self.local_size
                scale_min, scale_max = self.local_scale_aug

            scale = random.uniform(scale_min, scale_max)

            crop_h = max(1, int(round(out_size * scale)))
            crop_w = max(1, int(round(out_size * scale)))

            crop_h = min(crop_h, h)
            crop_w = min(crop_w, w)

            if h > crop_h:
                y1 = random.randint(0, h - crop_h)
            else:
                y1 = 0

            if w > crop_w:
                x1 = random.randint(0, w - crop_w)
            else:
                x1 = 0

            crop = img[y1:y1 + crop_h, x1:x1 + crop_w]

            if random.random() < 0.5:
                crop = cv2.flip(crop, 1)

            crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_CUBIC)

            crop = crop.astype(np.float32) / 255.0
            crop = np.expand_dims(crop, axis=0)   # [1, H, W]
            crop = torch.from_numpy(crop)

            crops.append(crop)

        return {
            "global_crops": crops[:2],
            "local_crops": crops[2:],
            "path": path
        }