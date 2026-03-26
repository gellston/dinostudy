import os
import glob
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
# skimage가 설치되어 있다면 사용하여 속도를 높일 수 있습니다. (pip install scikit-image)
# 여기서는 호환성을 위해 OpenCV/Numpy로만 구현합니다.

class DinoCropDataset(Dataset):
    def __init__(
        self,
        root_dir,
        global_size=224,
        local_size=96,
        # 중요: 미세 결함 학습을 위해 스케일 범위를 넓혀 모델에게 '어려운 퀴즈'를 줘야 합니다.
        global_scale_aug=(0.4, 1.0), 
        local_scale_aug=(0.05, 0.4),
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

    # --- 미세 결함용 맞춤형 증강 함수들 ---

    def _apply_grayscale_jitter(self, img, p=0.8):
        """밝기와 대비를 랜덤하게 조절하여 조명 변화에 대응합니다."""
        if random.random() < p:
            # 밝기(Brightness) 조절
            brightness = random.randint(-30, 30)
            img = cv2.add(img, brightness)
            
            # 대비(Contrast) 조절 (1.0 기준)
            contrast = random.uniform(0.8, 1.2)
            img = cv2.multiply(img, contrast)
            
            # 0~255 클리핑
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def _apply_gaussian_blur(self, img, p=0.5):
        """
        랜덤하게 블러를 적용합니다.
        중요: 파티클/스크래치가 지워지지 않도록 커널 사이즈를 매우 작게(3x3) 제한합니다.
        보통 Global View 1에만 강하게 적용하고, 나머지는 약하게 하거나 안 합니다.
        """
        if random.random() < p:
            # 작은 결함을 위해 커널은 3x3 고정
            img = cv2.GaussianBlur(img, (3, 3), 0)
        return img

    def _apply_gaussian_noise(self, img, p=0.3):
        """
        가우시안 노이즈를 추가합니다. 
        모델이 센서 노이즈와 진짜 파티클을 구분하는 능력을 기릅니다.
        """
        if random.random() < p:
            h, w = img.shape
            mean = 0
            # 노이즈 강도 조절 (너무 강하면 결함이 묻힘)
            sigma = random.uniform(2, 5) 
            gauss = np.random.normal(mean, sigma, (h, w)).astype('float32')
            
            img_f = img.astype('float32')
            noisy_img = cv2.add(img_f, gauss)
            img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        return img

    def _apply_solarization(self, img, p=0.2):
        """
        특정 임계값 이상의 밝기를 반전시킵니다. (DINO 정석 증강)
        금속 광택의 극단적인 반사를 시뮬레이션합니다.
        """
        if random.random() < p:
            # 임계값 (128 기준)
            threshold = 128
            img = np.where(img < threshold, img, 255 - img).astype(np.uint8)
        return img

    def _apply_basic_flip(self, img):
        """상하좌우 반전을 적용합니다. 제조업 데이터는 방향성이 없는 경우가 많습니다."""
        if random.random() < 0.5:
            img = cv2.flip(img, 1) # 좌우 반전
        if random.random() < 0.5:
            img = cv2.flip(img, 0) # 상하 반전 (파티클/스크래치 형태 불변)
        return img

    # -------------------------------------

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"이미지 로드 실패: {path}")

        h, w = img.shape[:2]
        global_crops = []
        g_coords = []

        # --- 1. Global View 2개 생성 ---
        for i in range(2):
            scale = random.uniform(self.global_scale_aug[0], self.global_scale_aug[1])
            # 크롭 크기 결정 (원본 이미지 비율 유지하며 크기만 조절)
            crop_h = min(h, max(1, int(round(self.global_size * scale))))
            crop_w = min(w, max(1, int(round(self.global_size * scale))))
            
            y1 = random.randint(0, h - crop_h) if h > crop_h else 0
            x1 = random.randint(0, w - crop_w) if w > crop_w else 0
            
            g_coords.append((y1, x1, crop_h, crop_w))
            
            # [A] 좌표 기반 크롭 (원본 픽셀 유지)
            crop = img[y1:y1 + crop_h, x1:x1 + crop_w]

            # [B] 미세 결함용 전용 증강 적용
            # 사용자님의 Horizontal Flip 로직을 _apply_basic_flip으로 확장
            crop = self._apply_basic_flip(crop)
            #crop = self._apply_grayscale_jitter(crop, p=0.8)
            crop = self._apply_gaussian_noise(crop, p=0.3) # 스크래치 vs 노이즈 구분용

            # DINO 논문 정석: Global 1에만 블러 적용, Global 2에는 솔라리제이션 적용
            #if i == 0:
            #    # 결함 안 지워지게 3x3으로 약하게
            #    crop = self._apply_gaussian_blur(crop, p=0.5) 
            #else:
            #    # 광택 반전 시뮬레이션
            #    crop = self._apply_solarization(crop, p=0.2) 
            
            # [C] 리사이즈 및 텐서화
            crop = cv2.resize(crop, (self.global_size, self.global_size), interpolation=cv2.INTER_CUBIC)
            crop = torch.from_numpy(crop.astype(np.float32) / 255.0).unsqueeze(0)
            global_crops.append(crop)

        # --- 2. Local View 생성 (글로벌 박스 내부 한정 로직 유지) ---
        local_crops = []
        for _ in range(self.local_crops_number):
            target_g_y1, target_g_x1, target_g_h, target_g_w = random.choice(g_coords)
            
            scale = random.uniform(self.local_scale_aug[0], self.local_scale_aug[1])
            l_crop_h = min(target_g_h, max(1, int(round(self.local_size * scale))))
            l_crop_w = min(target_g_w, max(1, int(round(self.local_size * scale))))
            
            y1 = target_g_y1 + (random.randint(0, target_g_h - l_crop_h) if target_g_h > l_crop_h else 0)
            x1 = target_g_x1 + (random.randint(0, target_g_w - l_crop_w) if target_g_w > l_crop_w else 0)
            
            # [A] 크롭
            crop = img[y1:y1 + l_crop_h, x1:x1 + l_crop_w]

            # [B] 로컬 뷰 전용 증강 (결함을 선명하게 유지하기 위해 약하게 적용)
            crop = self._apply_basic_flip(crop)
            # 밝기 조절은 배경과의 대비를 학습하기 위해 필요
            #crop = self._apply_grayscale_jitter(crop, p=0.8) 
            # 로컬 뷰에는 블러나 솔라리제이션을 거의 쓰지 않습니다. (결함 손상 방지)
            
            # [C] 리사이즈 및 텐서화
            crop = cv2.resize(crop, (self.local_size, self.local_size), interpolation=cv2.INTER_CUBIC)
            crop = torch.from_numpy(crop.astype(np.float32) / 255.0).unsqueeze(0)
            local_crops.append(crop)

        return {
            "global_crops": global_crops,
            "local_crops": local_crops,
            "path": path
        }