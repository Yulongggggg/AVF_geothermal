"""Conditional residual convolution block used by every encoder and decoder stage.

Source: ``temperature_revision/model.py`` (class ``CondBlock``), unchanged
arithmetic. See ``docs/PROVENANCE.md``.
"""
import torch
from torch import nn
from torch.nn import functional as F


class CondBlock(nn.Module):
    """Two 3x3 convolutions with group normalization, FiLM conditioning and a skip path.

    ``c`` is the 128-dimensional embedding of the five scalar controls. FiLM gains are
    passed through ``tanh`` so the modulation stays bounded.
    """

    def __init__(self, cin, cout, cdim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.film = nn.Linear(cdim, 2 * cout)
        nn.init.normal_(self.film.weight, std=.01)
        nn.init.zeros_(self.film.bias)

    def forward(self, x, c):
        h = F.gelu(self.norm1(self.conv1(x)))
        gain, shift = self.film(c).chunk(2, dim=1)
        h = h * (1 + torch.tanh(gain)[:, :, None, None]) + shift[:, :, None, None]
        h = self.norm2(self.conv2(F.gelu(h)))
        return F.gelu(h + self.skip(x))
