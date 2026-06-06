from __future__ import annotations
import torch
import torch.nn as nn
from .msaa import _MSAAChannelAttention, _MSAASpatialAttention

class SCAA(nn.Module):

    def __init__(self, channel: int | None=None, group: int=8, reduction: int=4) -> None:
        super().__init__()
        self.channel = channel
        self.group = int(group)
        self.reduction = int(reduction)
        self._built = False
        self._c = None
        self.cov1: nn.Module | None = None
        self.cov2: nn.Module | None = None
        self.cbam_list: nn.ModuleList | None = None

    def _build_if_needed(self, c: int) -> None:
        if self._built and self._c == c:
            return
        if c % self.group != 0:
            raise ValueError(f'SCAA: channel count {c} must be divisible by group count {self.group}')
        self.cov1 = nn.Conv2d(c, c, kernel_size=1)
        self.cov2 = nn.Conv2d(c, c, kernel_size=1)
        per = c // self.group
        modules = []
        for _ in range(self.group):
            ca = _MSAAChannelAttention(per, reduction=self.reduction)
            sa = _MSAASpatialAttention(kernel_size=7)
            modules.append(nn.Sequential(ca, sa))
        self.cbam_list = nn.ModuleList(modules)
        self._built = True
        self._c = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor) or x.dim() != 4:
            raise TypeError('SCAA expects input with shape [B, C, H, W]')
        (_, c, _, _) = x.shape
        self._build_if_needed(c)
        x0 = x
        x = self.cov1(x)
        chunks = torch.split(x, x.size(1) // self.group, dim=1)
        masks = []
        for (t, blk) in zip(chunks, self.cbam_list):
            m = blk(t)
            m = torch.sigmoid(m)
            mean = torch.mean(m, dim=(1, 2, 3), keepdim=True)
            gate = torch.ones_like(m) * mean
            mk = torch.where(m > gate, torch.ones_like(m), m)
            masks.append(mk.expand_as(t))
        mask = torch.cat(masks, dim=1)
        x = x * mask
        x = self.cov2(x)
        return x + x0
