"""Unlaunched numerical-repair candidate; frozen v2b remains unchanged.

Only the center focal calculation is promoted to FP32. The probability clamp,
normalization, loss weights and semantic/boundary terms retain the v2b formula.
Using this candidate in an experiment requires a new versioned plan.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from . import representation_train as base


def center_focal_fp32(logits, target, valid):
    with torch.amp.autocast(device_type=logits.device.type, enabled=False):
        logits = logits.float()
        target = target.float()
        valid = valid.float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite center logits; FP32 casting cannot repair them")
        probability = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
        positive = (target >= 1.0 - 1e-6).float() * valid
        negative = (target < 1.0 - 1e-6).float() * valid
        negative_weight = (1.0 - target).pow(4)
        positive_loss = -torch.log(probability) * (1.0 - probability).pow(2) * positive
        negative_loss = -torch.log(1.0 - probability) * probability.pow(2) * negative_weight * negative
        positive_count = positive.sum().clamp_min(1.0)
        valid_count = valid.sum().clamp_min(1.0)
        return positive_loss.sum() / positive_count + negative_loss.sum() / valid_count


def compute_loss_center_fp32(logits, targets, ignore):
    valid = (~ignore).float()
    semantic_logits, boundary_logits, center_logits = logits[:, 0], logits[:, 1], logits[:, 2]
    semantic_target, boundary_target, center_target = targets[:, 0], targets[:, 1], targets[:, 2]
    semantic_bce = base._masked_mean(
        F.binary_cross_entropy_with_logits(semantic_logits, semantic_target, reduction="none"), valid
    )
    semantic_probability = torch.sigmoid(semantic_logits) * valid
    semantic_target_valid = semantic_target * valid
    intersection = (semantic_probability * semantic_target_valid).sum(dim=(1, 2))
    denominator = semantic_probability.sum(dim=(1, 2)) + semantic_target_valid.sum(dim=(1, 2))
    semantic_dice_loss = 1.0 - ((2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()
    positive = (boundary_target * valid).sum()
    negative = valid.sum() - positive
    pos_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, 20.0)
    boundary_bce = F.binary_cross_entropy_with_logits(
        boundary_logits, boundary_target, reduction="none", pos_weight=pos_weight
    )
    center_focal = center_focal_fp32(center_logits, center_target, valid)
    return semantic_bce + semantic_dice_loss + 2.0 * base._masked_mean(boundary_bce, valid) + center_focal
