from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass
class ControlStats:
    contact_neg_fz_count: torch.Tensor  # (B,) int32
    fz_clamped_count: torch.Tensor      # (B,) int32
    min_z: torch.Tensor                 # (B,) float32
    first_contact_step: torch.Tensor    # (B,) int32, -1 if never


def make_control_stats(B: int, device: torch.device) -> ControlStats:
    return ControlStats(
        contact_neg_fz_count=torch.zeros(B, dtype=torch.int32, device=device),
        fz_clamped_count=torch.zeros(B, dtype=torch.int32, device=device),
        min_z=torch.full((B,), float("inf"), dtype=torch.float32, device=device),
        first_contact_step=torch.full((B,), -1, dtype=torch.int32, device=device),
    )


def table_safe_force(
    commanded: torch.Tensor,
    position: torch.Tensor,
    *,
    z_contact_eps: float = 1e-3,
    z_guard: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cancel downward force at floor contact; taper it in the guard band above."""
    z = position[..., 2]
    fz = commanded[..., 2]

    in_contact = z <= z_contact_eps
    in_guard = (z < z_guard) & ~in_contact
    fz_negative = fz < 0.0

    hard_clamp = in_contact & fz_negative
    soft_clamp = in_guard & fz_negative

    new_fz = fz.clone()
    new_fz = torch.where(hard_clamp, torch.zeros_like(new_fz), new_fz)

    if z_guard > 0.0:
        taper = torch.clamp(z / float(z_guard), min=0.0, max=1.0).pow(2)
        new_fz = torch.where(soft_clamp, new_fz * taper, new_fz)

    out = commanded.clone()
    out[..., 2] = new_fz
    return out, (hard_clamp | soft_clamp)


def apply_actuated_hemisphere_constraint(
    env_positions: torch.Tensor,
    env_velocities: torch.Tensor,
    actuate_idx: torch.Tensor,
    *,
    R: float = 0.9,
    z_min: float = 0.0,
) -> None:
    """Project actuated node back to hemisphere boundary if it exceeds reach limit R."""
    B = env_positions.shape[0]
    device = env_positions.device
    b_arange = torch.arange(B, device=device)

    p = env_positions[b_arange, actuate_idx].clone()
    v = env_velocities[b_arange, actuate_idx].clone()

    below_floor = p[..., 2] < z_min
    p[..., 2] = torch.where(below_floor, torch.full_like(p[..., 2], z_min), p[..., 2])
    v_z = v[..., 2]
    v[..., 2] = torch.where(below_floor & (v_z < 0), torch.zeros_like(v_z), v_z)

    r = torch.norm(p, dim=-1, keepdim=True)
    over = (r > R).squeeze(-1)
    if torch.any(over):
        r_safe = torch.clamp(r, min=1e-12)
        p = torch.where(over.unsqueeze(-1), R * p / r_safe, p)
        r_hat = p / torch.clamp(torch.norm(p, dim=-1, keepdim=True), min=1e-12)
        v_radial_mag = (v * r_hat).sum(dim=-1, keepdim=True)
        outward = (v_radial_mag.squeeze(-1) > 0) & over
        v = torch.where(outward.unsqueeze(-1), v - v_radial_mag * r_hat, v)

    env_positions[b_arange, actuate_idx] = p
    env_velocities[b_arange, actuate_idx] = v


def update_control_stats(
    stats: ControlStats,
    *,
    actuated_z: torch.Tensor,
    commanded_fz: torch.Tensor,
    fz_clamped_mask: torch.Tensor,
    step_idx: int,
    z_contact_eps: float,
) -> None:
    in_contact = actuated_z <= z_contact_eps
    bad_event = in_contact & (commanded_fz < 0.0)
    stats.contact_neg_fz_count += bad_event.to(torch.int32)
    stats.fz_clamped_count += fz_clamped_mask.to(torch.int32)
    stats.min_z = torch.minimum(stats.min_z, actuated_z)
    new_contact = in_contact & (stats.first_contact_step < 0)
    stats.first_contact_step = torch.where(
        new_contact,
        torch.full_like(stats.first_contact_step, int(step_idx)),
        stats.first_contact_step,
    )


def control_stats_to_meta_list(stats: ControlStats) -> List[dict]:
    return [
        {
            "contact_neg_fz_count": int(stats.contact_neg_fz_count[i].item()),
            "fz_clamped_count": int(stats.fz_clamped_count[i].item()),
            "min_z": float(stats.min_z[i].item()),
            "first_contact_step": int(stats.first_contact_step[i].item()),
        }
        for i in range(stats.contact_neg_fz_count.numel())
    ]
