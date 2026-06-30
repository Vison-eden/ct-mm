#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from utils import FEATS_NUM
except Exception:
    FEATS_NUM = tuple(f"feat_{i}" for i in range(14))


class TabEncoder(nn.Module):
    def __init__(
        self,
        feature_names: Sequence[str],
        d_model: int = 256,
        dropout: float = 0.2,
        density_num_classes: int = 4,
    ) -> None:
        super().__init__()

        del density_num_classes

        self.feature_names = list(feature_names)
        self.n_features = len(self.feature_names)

        self.net = nn.Sequential(
            nn.Linear(self.n_features + 1, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        x_num: torch.Tensor,
        x_density: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        x_density = x_density.float().view(-1, 1)
        x = torch.cat([x_num.float(), x_density], dim=1)

        out = self.net(x)

        return out.unsqueeze(1), out


class ResNet3DBackbone(nn.Module):
    def __init__(
        self,
        name: str = "r2plus1d_18",
        in_channels: int = 1,
    ) -> None:
        super().__init__()

        del name

        self.out_dim = 128

        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),

            nn.Conv3d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),

            nn.Conv3d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool3d(1),
        )

        self.proj = nn.Linear(64, self.out_dim)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        feat = self.net(x)
        pooled = feat.flatten(1)
        pooled = self.proj(pooled)

        return pooled, feat


class ImageEncoder3D(nn.Module):
    def __init__(
        self,
        backbone: str = "r2plus1d_18",
        in_channels: int = 1,
        d_model: int = 256,
        transformer_layers: int = 0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        del d_model, transformer_layers, dropout

        self.backbone = ResNet3DBackbone(
            name=backbone,
            in_channels=in_channels,
        )

        self.out_dim = self.backbone.out_dim

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        pooled, feat = self.backbone(x)

        return pooled, pooled.unsqueeze(1)


class FusionHead(nn.Module):
    def __init__(
        self,
        d_img: int,
        d_tab: int,
        out_classes: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        del dropout

        self.classifier = nn.Linear(d_img + d_tab, out_classes)

    def forward(
        self,
        g_img: torch.Tensor,
        g_tab: torch.Tensor,
    ) -> torch.Tensor:

        x = torch.cat([g_img, g_tab], dim=1)

        return self.classifier(x)


class BiCrossFusion(nn.Module):
    def __init__(
        self,
        d_img: int,
        d_tab: int,
        out_classes: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.head = FusionHead(
            d_img=d_img,
            d_tab=d_tab,
            out_classes=out_classes,
            dropout=dropout,
        )

    def forward(
        self,
        g_img: torch.Tensor,
        tok_img: torch.Tensor,
        g_tab: torch.Tensor,
        tok_tab: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        del tok_img, tok_tab

        logits = self.head(g_img, g_tab)

        z_img = torch.empty(
            g_img.size(0),
            0,
            device=g_img.device,
            dtype=g_img.dtype,
        )

        z_tab = torch.empty(
            g_tab.size(0),
            0,
            device=g_tab.device,
            dtype=g_tab.dtype,
        )

        return logits, z_img, z_tab


class GatingHead(nn.Module):
    def __init__(
        self,
        d_img: int,
        d_tab: int,
        density_num_classes: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        del d_img, d_tab, density_num_classes, dropout

    def forward(
        self,
        g_img: torch.Tensor,
        g_tab: torch.Tensor,
        density_id: torch.Tensor,
        size_z: torch.Tensor,
    ) -> torch.Tensor:

        del g_tab, density_id, size_z

        return torch.full(
            (g_img.size(0), 3),
            1.0 / 3.0,
            device=g_img.device,
            dtype=g_img.dtype,
        )


class ImgOnlyModel(nn.Module):
    def __init__(
        self,
        img_backbone: str = "r2plus1d_18",
        in_ch: int = 1,
        d_img: int = 256,
        drop: float = 0.2,
        img_trans_layers: int = 0,
        vit_patch: Optional[Tuple[int, int, int]] = None,
        vit_depth: int = 2,
        out_classes: int = 3,
    ) -> None:
        super().__init__()

        del d_img, drop, img_trans_layers, vit_patch, vit_depth

        self.out_classes = int(out_classes)

        self.img = ImageEncoder3D(
            backbone=img_backbone,
            in_channels=in_ch,
        )

        self.head = nn.Linear(
            self.img.out_dim,
            self.out_classes,
        )

    def forward(
        self,
        x_img: torch.Tensor,
        _tab=None,
        _ld_z=None,
        _den_id=None,
    ):

        g_img, _ = self.img(x_img)
        logits = self.head(g_img)

        if self.out_classes == 1:
            logits = logits.squeeze(1)

        return logits, {}


class TabOnlyModel(nn.Module):
    def __init__(
        self,
        d_txt: int = 256,
        drop: float = 0.2,
        den_num_classes: int = 4,
        out_classes: int = 3,
        num_names: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()

        names = list(num_names) if num_names is not None else list(FEATS_NUM)

        self.out_classes = int(out_classes)

        self.txt = TabEncoder(
            feature_names=names,
            d_model=d_txt,
            dropout=drop,
            density_num_classes=den_num_classes,
        )

        self.head = nn.Linear(
            d_txt,
            self.out_classes,
        )

    def forward(
        self,
        _img,
        x_tab,
        _ld_z=None,
        _den_id=None,
    ):

        x_num, x_density = x_tab

        _, g_tab = self.txt(x_num, x_density)
        logits = self.head(g_tab)

        if self.out_classes == 1:
            logits = logits.squeeze(1)

        return logits, {}


class UnifiedModel(nn.Module):
    def __init__(
        self,
        mode: str = "mm",
        task_out_classes: int = 3,
        img_backbone: str = "r2plus1d_18",
        in_ch: int = 1,
        img_trans_layers: int = 0,
        d_img: int = 256,
        d_txt: int = 256,
        den_num_classes: int = 4,
        drop: float = 0.2,
        use_gate: bool = False,
        mix_rule: str = "prob_anchor",
        alpha_cap: float = 0.30,
        vit_patch: Optional[Tuple[int, int, int]] = None,
        vit_depth: int = 2,
        num_names: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()

        del mode
        del img_trans_layers
        del d_img
        del use_gate
        del mix_rule
        del alpha_cap
        del vit_patch
        del vit_depth

        self.out_classes = int(task_out_classes)

        names = list(num_names) if num_names is not None else list(FEATS_NUM)

        self.img = ImageEncoder3D(
            backbone=img_backbone,
            in_channels=in_ch,
        )

        self.txt = TabEncoder(
            feature_names=names,
            d_model=d_txt,
            dropout=drop,
            density_num_classes=den_num_classes,
        )

        self.head_img = nn.Linear(
            self.img.out_dim,
            self.out_classes,
        )

        self.head_txt = nn.Linear(
            d_txt,
            self.out_classes,
        )

        self.fuse = FusionHead(
            d_img=self.img.out_dim,
            d_tab=d_txt,
            out_classes=self.out_classes,
            dropout=drop,
        )

    def forward(
        self,
        x_img: torch.Tensor,
        x_tab,
        ld_z: torch.Tensor,
        den_id: torch.Tensor,
    ):

        del ld_z, den_id

        x_num, x_density = x_tab

        g_img, tok_img = self.img(x_img)
        tok_tab, g_tab = self.txt(x_num, x_density)

        del tok_img, tok_tab

        logits_img = self.head_img(g_img)
        logits_txt = self.head_txt(g_tab)
        logits = self.fuse(g_img, g_tab)

        if self.out_classes == 1:
            logits = logits.squeeze(1)

        extra = {
            "logits_img": logits_img,
            "logits_txt": logits_txt,
        }

        return logits, extra
