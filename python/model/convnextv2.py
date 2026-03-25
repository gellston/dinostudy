import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if mean < a - 2 * std or mean > b + 2 * std:
        warnings.warn("mean is more than 2 std from [a, b] in trunc_normal_.")

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()

        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        tensor.clamp_(min=a, max=b)
        return tensor
    




class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)

        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)

        return x * random_tensor

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


class Block(nn.Module):
    """ConvNeXtV2 Block (NCHW, 1x1 Conv, GRN2d version)"""
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.grn = GRN2d(4 * dim)
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = shortcut + self.drop_path(x)
        return x


class ConvNeXtV2(nn.Module):
    def __init__(
        self,
        in_chans=3,
        num_classes=1,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.0,
        head_init_scale=1.0,
    ):
        super().__init__()
        self.depths = depths

        self.downsample_layers = nn.ModuleList()

        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0], eps=1e-6),
        )
        self.downsample_layers.append(self.stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm2d(dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = x.mean([-2, -1])   # global average pooling: (N, C, H, W) -> (N, C)
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def convnextv2_atto(**kwargs):
    return ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)


def convnextv2_femto(**kwargs):
    return ConvNeXtV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)


def convnextv2_pico(**kwargs):
    return ConvNeXtV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)


def convnextv2_nano(**kwargs):
    return ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)


def convnextv2_tiny(**kwargs):
    return ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)


def convnextv2_base(**kwargs):
    return ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)


def convnextv2_large(**kwargs):
    return ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)


def convnextv2_huge(**kwargs):
    return ConvNeXtV2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)