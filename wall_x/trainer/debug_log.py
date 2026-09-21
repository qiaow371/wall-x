"""Train debug log helpers (TB/wandb scalars from unreduced flow MSE).

See data_Driven docs/TRAIN_DEBUG_LOG.md: split flow vs AR CE, EE 20-D per-dim,
groups, per-step, pad, NaN. Do not mean() before slicing.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

JOINT_16 = [f"L{j}" for j in range(1, 8)] + [f"R{j}" for j in range(1, 8)] + ["Lgrip", "Rgrip"]
EE_20 = (
    ["L_x", "L_y", "L_z"]
    + [f"L_r6d{i}" for i in range(6)]
    + ["L_grip"]
    + ["R_x", "R_y", "R_z"]
    + [f"R_r6d{i}" for i in range(6)]
    + ["R_grip"]
)


def dim_names(d: int) -> list[str]:
    if d == 16:
        return list(JOINT_16)
    if d >= 20:
        names = list(EE_20)
        if d > 20:
            names.extend(f"pad{i}" for i in range(d - 20))
        return names[:d]
    return [f"d{i}" for i in range(d)]


def _masked_mean(loss: torch.Tensor, valid: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    num = (loss * valid).sum(dim=dims)
    den = valid.sum(dim=dims).clamp(min=1.0)
    return num / den


def flow_debug_stats(
    flow_btd: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Slice unreduced flow MSE [B, T, D] into cheap scalars (detached)."""
    loss = flow_btd.detach().float()
    if valid_mask is None:
        valid = torch.ones_like(loss)
    else:
        valid = valid_mask.detach().float()
        if valid.shape != loss.shape:
            valid = valid.expand_as(loss)

    stats: Dict[str, torch.Tensor] = {}
    b, t, d = loss.shape
    names = dim_names(d)
    per_dim = _masked_mean(loss, valid, (0, 1))
    per_step = _masked_mean(loss, valid, (0, 2))

    for i, name in enumerate(names):
        stats[f"Action loss dim/{name}"] = per_dim[i]
    if t > 0:
        stats["Action loss per_step/head"] = per_step[0]
        stats["Action loss per_step/tail"] = per_step[-1]
        stats["Action loss per_step/tail_head_ratio"] = per_step[-1] / per_step[0].clamp(min=1e-8)
        for i in range(t):
            stats[f"Action loss per_step/{i}"] = per_step[i]

    if d >= 20:
        xyz = torch.cat([per_dim[0:3], per_dim[10:13]])
        r6d = torch.cat([per_dim[3:9], per_dim[13:19]])
        grip = torch.stack([per_dim[9], per_dim[19]])
        stats["Action loss group/xyz"] = xyz.mean()
        stats["Action loss group/r6d"] = r6d.mean()
        stats["Action loss group/grip"] = grip.mean()
        if d > 20:
            pad = per_dim[20:]
            stats["training/pad_mean"] = pad.mean()
            stats["Action loss group/pad"] = pad.mean()
    elif d == 16:
        stats["Action loss group/L_joints"] = per_dim[0:7].mean()
        stats["Action loss group/R_joints"] = per_dim[7:14].mean()
        stats["Action loss group/grip"] = per_dim[14:16].mean()
    else:
        stats["Action loss group/all"] = per_dim.mean()

    finite = torch.isfinite(loss)
    stats["training/nan"] = (~finite).float().mean()
    stats["training/flow_unreduced_mean"] = _masked_mean(loss, valid, (0, 1, 2))
    return stats
