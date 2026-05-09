#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model definitions for multimodal lung nodule classification."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import FEATS_NUM


class TabEncoder(nn.Module):
    def __init__(
        self,
        feature_names: Sequence[str],
        d_model: int = 256,
        dropout: float = 0.2,
        density_num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.feature_names = list(feature_names)
        self.name_embedding = nn.Embedding(len(self.feature_names), d_model)
        self.value_projection = nn.Linear(1, d_model)
        self.density_embedding = nn.Embedding(density_num_classes, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(
        self,
        x_num: torch.Tensor,
        x_density: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, n_features = x_num.shape

        if n_features != len(self.feature_names):
            raise RuntimeError(
                f"Expected {len(self.feature_names)} structured features, "
                f"got {n_features}. Expected order: {self.feature_names}"
            )

        idx = torch.arange(n_features, device=x_num.device)
        name_tokens = self.name_embedding(idx).unsqueeze(0)
        value_tokens = self.value_projection(x_num.unsqueeze(-1))
        num_tokens = value_tokens + name_tokens
        density_token = self.density_embedding(x_density.long()).unsqueeze(1)

        cls = self.cls_token.expand(batch_size, 1, -1)
        seq = torch.cat([cls, num_tokens, density_token], dim=1)
        seq = self.encoder(seq)

        return seq[:, 1:, :], seq[:, 0, :]


class ResNet3DBackbone(nn.Module):
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

        net = builders[name](weights=None)

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
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

            if old_conv.weight.shape[1] == 3 and in_channels == 1:
                with torch.no_grad():
                    new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))

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
    def __init__(
        self,
        backbone: str = "r2plus1d_18",
        in_channels: int = 1,
        d_model: int = 256,
        transformer_layers: int = 0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone = ResNet3DBackbone(backbone, in_channels)
        self.transformer_layers = int(transformer_layers)

        if self.transformer_layers > 0:
            self.token_projection = nn.Linear(self.backbone.out_dim, d_model)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=512,
                dropout=dropout,
                batch_first=True,
            )
            self.token_encoder = nn.TransformerEncoder(
                layer,
                num_layers=self.transformer_layers,
            )
            nn.init.normal_(self.cls_token, std=0.02)
            self.out_dim = d_model
        else:
            self.out_dim = self.backbone.out_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled, feat = self.backbone(x)
        tokens = feat.flatten(2).transpose(1, 2)

        if self.transformer_layers <= 0:
            return pooled, tokens

        batch_size = x.size(0)
        tokens = self.token_projection(tokens)
        cls = self.cls_token.expand(batch_size, 1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        seq = self.token_encoder(seq)

        return seq[:, 0, :], seq[:, 1:, :]


class BiCrossFusion(nn.Module):
    def __init__(
        self,
        d_img: int,
        d_tab: int,
        out_classes: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.query_img_to_tab = nn.Linear(d_img, d_tab)
        self.query_tab_to_img = nn.Linear(d_tab, d_img)

        self.attn_tab = nn.MultiheadAttention(
            d_tab,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_img = nn.MultiheadAttention(
            d_img,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )

        self.gate_tab = nn.Sequential(
            nn.Linear(d_img, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.gate_img = nn.Sequential(
            nn.Linear(d_tab, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.log_temp_tab = nn.Parameter(torch.zeros(()))
        self.log_temp_img = nn.Parameter(torch.zeros(()))

        self.head = nn.Sequential(
            nn.Linear(d_img + d_tab + d_img + d_tab, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, out_classes),
        )

        self.proj_img = nn.Linear(d_img, 128)
        self.proj_tab = nn.Linear(d_tab, 128)

    def forward(
        self,
        g_img: torch.Tensor,
        tok_img: torch.Tensor,
        g_tab: torch.Tensor,
        tok_tab: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        temp_tab = torch.exp(self.log_temp_tab).clamp(0.5, 2.0)
        temp_img = torch.exp(self.log_temp_img).clamp(0.5, 2.0)

        q_tab = (self.query_img_to_tab(g_img) / temp_tab).unsqueeze(1)
        q_img = (self.query_tab_to_img(g_tab) / temp_img).unsqueeze(1)

        ctx_tab, _ = self.attn_tab(q_tab, tok_tab, tok_tab)
        ctx_img, _ = self.attn_img(q_img, tok_img, tok_img)

        ctx_tab = ctx_tab.squeeze(1) * self.gate_tab(g_img)
        ctx_img = ctx_img.squeeze(1) * self.gate_img(g_tab)

        fused = torch.cat([g_img, g_tab, ctx_img, ctx_tab], dim=1)
        logits = self.head(fused)

        z_img = F.normalize(self.proj_img(g_img), dim=1, eps=1e-8)
        z_tab = F.normalize(self.proj_tab(g_tab), dim=1, eps=1e-8)

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
        self.density_embedding = nn.Embedding(density_num_classes, 16)
        self.net = nn.Sequential(
            nn.Linear(d_img + d_tab + 16 + 1, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )

    def forward(
        self,
        g_img: torch.Tensor,
        g_tab: torch.Tensor,
        density_id: torch.Tensor,
        size_z: torch.Tensor,
    ) -> torch.Tensor:
        density = self.density_embedding(density_id.long())
        x = torch.cat([g_img, g_tab, density, size_z.unsqueeze(1)], dim=1)
        return F.softmax(self.net(x), dim=1)


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

        self.out_classes = out_classes
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
            nn.Linear(256, out_classes),
        )

    def forward(self, x_img: torch.Tensor, _tab=None, _ld_z=None, _den_id=None):
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
        self.out_classes = out_classes
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
            nn.Linear(256, out_classes),
        )

    def forward(self, _img, x_tab, _ld_z=None, _den_id=None):
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
        del mode, vit_patch, vit_depth

        self.out_classes = task_out_classes
        self.use_gate = bool(use_gate)
        self.mix_rule = mix_rule
        self.alpha_cap = float(alpha_cap)

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
            nn.Linear(256, task_out_classes),
        )
        self.head_txt = nn.Sequential(
            nn.Linear(d_txt, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(256, task_out_classes),
        )
        self.fuse = BiCrossFusion(
            d_img=self.img.out_dim,
            d_tab=d_txt,
            out_classes=task_out_classes,
            dropout=drop,
        )

        if self.use_gate:
            self.gate = GatingHead(
                d_img=self.img.out_dim,
                d_tab=d_txt,
                density_num_classes=den_num_classes,
                dropout=drop,
            )

    def forward(
        self,
        x_img: torch.Tensor,
        x_tab,
        ld_z: torch.Tensor,
        den_id: torch.Tensor,
    ):
        x_num, x_density = x_tab

        g_img, tok_img = self.img(x_img)
        tok_tab, g_tab = self.txt(x_num, x_density)

        logits_img = self.head_img(g_img)
        logits_txt = self.head_txt(g_tab)
        logits_fuse, z_img, z_tab = self.fuse(g_img, tok_img, g_tab, tok_tab)

        extra = {
            "logits_img": logits_img,
            "logits_txt": logits_txt,
            "pi": z_img,
            "pt": z_tab,
        }

        if not self.use_gate:
            if self.out_classes == 1:
                return logits_fuse.squeeze(1), extra
            return logits_fuse, extra

        p_gate = self.gate(g_img, g_tab, den_id, ld_z)
        extra["p_gate"] = p_gate

        eps = 1e-6

        if self.out_classes == 1:
            p_img = torch.sigmoid(logits_img).squeeze(1).clamp(eps, 1.0 - eps)
            p_txt = torch.sigmoid(logits_txt).squeeze(1).clamp(eps, 1.0 - eps)
            p_fuse = torch.sigmoid(logits_fuse).squeeze(1).clamp(eps, 1.0 - eps)

            if self.mix_rule == "prob_anchor":
                alpha = self.alpha_cap * (1.0 - p_gate[:, 2])
                denom = p_gate[:, 0] + p_gate[:, 1] + eps
                w_txt = p_gate[:, 0] / denom
                w_img = p_gate[:, 1] / denom
                p = (1.0 - alpha) * p_fuse + alpha * (w_img * p_img + w_txt * p_txt)
                p = p.clamp(eps, 1.0 - eps)
                return torch.log(p) - torch.log1p(-p), extra

            logits = (
                p_gate[:, 0] * logits_txt.squeeze(1)
                + p_gate[:, 1] * logits_img.squeeze(1)
                + p_gate[:, 2] * logits_fuse.squeeze(1)
            )
            return logits, extra

        if self.mix_rule == "prob_anchor":
            p_img = F.softmax(logits_img, dim=1).clamp_min(eps)
            p_txt = F.softmax(logits_txt, dim=1).clamp_min(eps)
            p_fuse = F.softmax(logits_fuse, dim=1).clamp_min(eps)

            alpha = self.alpha_cap * (1.0 - p_gate[:, 2]).unsqueeze(1)
            denom = (p_gate[:, 0] + p_gate[:, 1] + eps).unsqueeze(1)
            w_txt = p_gate[:, 0].unsqueeze(1) / denom
            w_img = p_gate[:, 1].unsqueeze(1) / denom

            p = (1.0 - alpha) * p_fuse + alpha * (w_img * p_img + w_txt * p_txt)
            return torch.log(p.clamp_min(eps)), extra

        logits = (
            p_gate[:, 0:1] * logits_txt
            + p_gate[:, 1:2] * logits_img
            + p_gate[:, 2:3] * logits_fuse
        )
        return logits, extra