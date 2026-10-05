"""Verbatim scientific definitions extracted from pinned original source."""
import torch
import torch.nn.functional as F
H,W=160,128

def field_on_grid(t, span_y=400.0):
    """(B,C,n,n) 覆盖 [-400,400]² 的方形场 → (B,C,H,W) 计算网格(y∈[-500,500],缓冲带复制边缘)。"""
    yq = torch.linspace(-500, 500, H, device=t.device).clamp(-span_y, span_y)/span_y
    xq = torch.linspace(-400, 400, W, device=t.device)/400.0
    gy, gx = torch.meshgrid(yq, xq, indexing="ij")
    grid = torch.stack([gx, gy], -1)[None].expand(t.shape[0], -1, -1, -1).to(t.dtype)
    return F.grid_sample(t, grid, mode="bilinear", align_corners=True)
