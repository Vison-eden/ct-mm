#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Public model definitions for multimodal lung nodule classification.

This release keeps the basic model interfaces for readability and compatibility,
while using simplified baseline components. Experimental fusion and gating
details are intentionally omitted from the public version.
"""

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
    """
    Simplified tabular encoder.

    Public version:
    - uses a plain MLP over structured features and density embedding;
    - does not expose feature-token modeling or feature-order-specific design.
    """

    def __init__(
        self,
        feature_names: Sequence[str],
        d_model: int = 256,
        dropout: float = 0.2,
        density_num_classes: int = 4,
    ) -> None:
        super().__init__()

        self.feature_names = list(feature_names)
        self.n_features = len(self.feature_names)
        self.density_num_classes = int(density_num_classes)

        self.density_embedding = nn.Embedding(self.density_num_classes, 16)

        self.encoder = nn.Sequential(
            nn.Linear(self.n_features + 16, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        x_num: torch.Tensor,
        x_density: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        x_density = x_density.long().view(-1)
        x_density = x_density.clamp(0, self.density_num_classes - 1)

        density = self.density_embedding(x_density)
        x = torch.cat([x_num.float(), density], dim=1)

        g_tab = self.encoder(x)

        # Keep a token-like output for interface compatibility.
        tok_tab = g_tab.unsqueeze(1)

        return tok_tab, g_tab


class ResNet3DBackbone(nn.Module):
    """
    Standard 3D CNN backbone.

    Public version keeps only the common torchvision video backbones.
    """

    def __init__(self, name: str = "r2plus1d_18", in_channels: int = 1) -> None:
        super().__init__()

        import torchvision.models.video as video_models

        builders = {
            "r3d_18": video_models.r3d_18,
            "mc3_18": video_models.mc3_18,
            "r2plus1d_18": video_models.r2plus1d_18,
        }

        if name not in builders:
            raise ValueError(f"Unsupported 3D backbone: {name}")

        try:
            net = builders[name](weights=None)
        except TypeError:
            net = builders[name](pretrained=False)

        old_conv = net.stem[0]

        if old_conv.in_channels != in_channels:
            new_conv = nn.Conv3d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )

            nn.init.kaiming_normal_(
                new_conv.weight,
                mode="fan_out",
                nonlinearity="relu",
            )

            if old_conv.weight.shape[1] == 3 and in_channels == 1:
                with torch.no_grad():
                    new_conv.weight.copy_(
                        old_conv.weight.mean(dim=1, keepdim=True)
                    )

            net.stem[0] = new_conv

        self.net = net
        self.out_dim = net.fc.in_features
        self.net.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net.stem(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        feat = self.net.layer4(x)

        pooled = F.adaptive_avg_pool3d(feat, 1).flatten(1)

        return pooled, feat


class ImageEncoder3D(nn.Module):
    """
    Simplified image encoder.

    Public version:
    - uses 3D CNN global pooled features;
    - omits additional token-transformer refinement.
    """

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

        self.backbone = ResNet3DBackbone(backbone, in_channels)
        self.out_dim = self.backbone.out_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled, _feat = self.backbone(x)

        # Keep a token-like output for interface compatibility.
        tokens = pooled.unsqueeze(1)

        return pooled, tokens


class FusionHead(nn.Module):
    """
    Public baseline fusion head.

    This is a simple concatenation-based fusion module.
    """

    def __init__(
        self,
        d_img: int,
        d_tab: int,
        out_classes: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(d_img + d_tab, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, out_classes),
        )

    def forward(
        self,
        g_img: torch.Tensor,
        g_tab: torch.Tensor,
    ) -> torch.Tensor:

        x = torch.cat([g_img, g_tab], dim=1)
        logits = self.classifier(x)

        return logits


class BiCrossFusion(nn.Module):
    """
    Compatibility wrapper.

    The original experimental fusion implementation is not included in the
    public release. This class now behaves as a plain concatenation baseline.
    """

    def __init__(
        self,
        d_img: int,
        d_tab: int,
        out_classes: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.fusion = FusionHead(
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

        logits = self.fusion(g_img, g_tab)

        # Empty placeholders for compatibility only.
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
    """
    Compatibility placeholder.

    Dynamic gating details are not included in this public release.
    """

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

        batch_size = g_img.size(0)
        weights = torch.full(
            (batch_size, 3),
            1.0 / 3.0,
            device=g_img.device,
            dtype=g_img.dtype,
        )

        return weights


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

        del vit_patch, vit_depth

        self.out_classes = int(out_classes)

        self.img = ImageEncoder3D(
            backbone=img_backbone,
            in_channels=in_ch,
            d_model=d_img,
            transformer_layers=img_trans_layers,
            dropout=drop,
        )

        self.head = nn.Sequential(
            nn.Linear(self.img.out_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(256, self.out_classes),
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
            names,
            d_model=d_txt,
            dropout=drop,
            density_num_classes=den_num_classes,
        )

        self.head = nn.Sequential(
            nn.Linear(d_txt, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(256, self.out_classes),
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
    """
    Public multimodal model.

    Public version:
    - image branch: 3D CNN encoder;
    - tabular branch: MLP encoder;
    - fusion branch: concatenation classifier;
    - omitted: experimental cross-attention, adaptive gating, probability-anchor
      fusion, and contrastive auxiliary heads.
    """

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

        del mode, use_gate, mix_rule, alpha_cap, vit_patch, vit_depth

        self.out_classes = int(task_out_classes)

        names = list(num_names) if num_names is not None else list(FEATS_NUM)

        self.img = ImageEncoder3D(
            backbone=img_backbone,
            in_channels=in_ch,
            d_model=d_img,
            transformer_layers=img_trans_layers,
            dropout=drop,
        )

        self.txt = TabEncoder(
            names,
            d_model=d_txt,
            dropout=drop,
            density_num_classes=den_num_classes,
        )

        self.head_img = nn.Sequential(
            nn.Linear(self.img.out_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(256, self.out_classes),
        )

        self.head_txt = nn.Sequential(
            nn.Linear(d_txt, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(256, self.out_classes),
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

        g_img, _tok_img = self.img(x_img)
        _tok_tab, g_tab = self.txt(x_num, x_density)

        logits_img = self.head_img(g_img)
        logits_txt = self.head_txt(g_tab)
        logits_fuse = self.fuse(g_img, g_tab)

        extra = {
            "logits_img": logits_img,
            "logits_txt": logits_txt,
        }

        if self.out_classes == 1:
            logits_fuse = logits_fuse.squeeze(1)

        return logits_fuse, extra
