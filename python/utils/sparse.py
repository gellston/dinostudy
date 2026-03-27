import torch
import torch.nn as nn


_cur_active: torch.Tensor = None # B1ff

def update_cur_active(input:torch.Tensor, prob):
    b = input.shape[0]
    h = input.shape[2]
    w = input.shape[3]

    make_cur_active(b, h, w, prob, device=input.device)

def make_cur_active(B, H, W, prob, device=None):
    """
    _cur_active 생성 함수

    Args:
        B (int): batch size
        H (int): height (mask grid)
        W (int): width (mask grid)
        prob (float): 각 위치가 1이 될 확률 (0~1)
        device: torch device

    Returns:
        torch.Tensor: (B, 1, H, W) binary mask
    """
    if device is None:
        device = torch.device("cpu")

    mask = (torch.rand(B, 1, H, W, device=device) < prob).float()

    global _cur_active
    _cur_active = mask
    return mask


def _get_active_ex_or_ii(H, W, returning_active_ex=True):
    global _cur_active

    active_ex = torch.nn.functional.interpolate(
        _cur_active.float(),
        size=(H, W),
        mode="nearest"
    )

    if returning_active_ex:
        return active_ex
    else:
        return active_ex.squeeze(1).nonzero(as_tuple=True)


# def _get_active_ex_or_ii(H, W, returning_active_ex=True):
#     h_repeat, w_repeat = H // _cur_active.shape[-2], W // _cur_active.shape[-1]
#     active_ex = _cur_active.repeat_interleave(h_repeat, dim=2).repeat_interleave(w_repeat, dim=3)
#     return active_ex if returning_active_ex else active_ex.squeeze(1).nonzero(as_tuple=True)  # ii: bi, hi, wi


def sp_conv_forward(self, x: torch.Tensor):
    x = super(type(self), self).forward(x)
    mask =  _get_active_ex_or_ii(H=x.shape[2], W=x.shape[3], returning_active_ex=True)    # (BCHW) *= (B1HW), mask the output of conv
    x *= mask
    return x


def sp_bn_forward(self, x: torch.Tensor):
    ii = _get_active_ex_or_ii(H=x.shape[2], W=x.shape[3], returning_active_ex=False)
    
    bhwc = x.permute(0, 2, 3, 1)
    nc = bhwc[ii]                               # select the features on non-masked positions to form a flatten feature `nc`
    nc = super(type(self), self).forward(nc)    # use BN1d to normalize this flatten feature `nc`
    
    bchw = torch.zeros_like(bhwc)
    bchw[ii] = nc
    bchw = bchw.permute(0, 3, 1, 2)
    return bchw


def sp_ln_forward(self, x):
    ii = _get_active_ex_or_ii(H=x.shape[2], W=x.shape[3], returning_active_ex=False)

    # (B,C,H,W) → (B,H,W,C)
    bhwc = x.permute(0, 2, 3, 1)

    # active 위치만 선택
    nc = bhwc[ii]              # (N_active, C)

    # LayerNorm 적용
    nc = self.norm(nc)

    # 다시 원래 위치에 복원
    out = torch.zeros_like(bhwc)
    out[ii] = nc

    # (B,C,H,W)로 복구
    out = out.permute(0, 3, 1, 2)
    return out


def sp_grn_forward(self, x):
    mask = _get_active_ex_or_ii(H=x.shape[2], W=x.shape[3], returning_active_ex=True)

    # mask 적용
    x_masked = x * mask

    # L2 norm (masked)
    gx = torch.norm(x_masked, p=2, dim=(2, 3), keepdim=True)

    # normalization (channel-wise)
    nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)

    return self.gamma * (x * nx) + self.beta + x


class SparseConv2d(nn.Conv2d):
    forward = sp_conv_forward   # hack: override the forward function; see `sp_conv_forward` above for more details


class SparseMaxPooling(nn.MaxPool2d):
    forward = sp_conv_forward   # hack: override the forward function; see `sp_conv_forward` above for more details


class SparseAvgPooling(nn.AvgPool2d):
    forward = sp_conv_forward   # hack: override the forward function; see `sp_conv_forward` above for more details


class SparseBatchNorm2d(nn.BatchNorm1d):
    forward = sp_bn_forward     # hack: override the forward function; see `sp_bn_forward` above for more details


class SparseSyncBatchNorm2d(nn.SyncBatchNorm):
    forward = sp_bn_forward     # hack: override the forward function; see `sp_bn_forward` above for more details

class SparseLayerNorm2d(nn.Module):
    forward = sp_ln_forward
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)
        

class SparseGRN2d(nn.Module):
    forward = sp_grn_forward
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.eps = eps
