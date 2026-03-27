

import torch
import torch.nn as nn

import cv2
import numpy as np

def show_image(name, width, height, tensor):
    """
    tensor: (B, 1, H, W) or (B, C, H, W)
    """

    # 1. 첫 번째 이미지 선택
    img = tensor[0]  # (1, H, W)

    # 2. CPU로 이동 + numpy 변환
    img = img.detach().cpu().numpy()

    # 3. 채널 제거 (1, H, W) → (H, W)
    img = img.squeeze()

    # 4. 정규화 (0~1 → 0~255)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = (img * 255).astype(np.uint8)

    # 5. OpenCV로 출력
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, width, height)
    cv2.imshow(name, img)



def copy_weights_ignore_name(src_model, dst_model):
    src_params = list(src_model.parameters())
    dst_params = list(dst_model.parameters())

    assert len(src_params) == len(dst_params), \
        f"Parameter count mismatch: {len(src_params)} vs {len(dst_params)}"

    for src, dst in zip(src_params, dst_params):
        assert src.shape == dst.shape, \
            f"Shape mismatch: {src.shape} vs {dst.shape}"
        dst.data.copy_(src.data)
        


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        # x: (N, C, H, W)
        x = x.permute(0, 2, 3, 1)   # -> (N, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)   # -> (N, C, H, W)
        return x


class GRN2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.eps = eps

    def forward(self, x):
        # x: (N, C, H, W)
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)              # (N, C, 1, 1)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)            # (N, C, 1, 1)
        return self.gamma * (x * nx) + self.beta + x