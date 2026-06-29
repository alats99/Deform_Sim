from __future__ import annotations

import math
import random
from typing import Any, Dict, Tuple, Union

import torch

__all__ = [
    "generate_horizontal",
    "generate_hanging",
    "generate_coiled_spiral",
    "generate_random_walk",
    "build_init_pose",
]


def _random_walk(N: int, length: float, gen: torch.Generator) -> torch.Tensor:
    positions = torch.zeros((N, 3))
    segment = length / (N - 1)
    for i in range(1, N):
        d = torch.randn((3,), generator=gen)
        d = d / torch.clamp(torch.norm(d), min=1e-8)
        positions[i] = positions[i - 1] + d * segment
    return positions


def _sample_anchor_in_hemisphere(
    rng: random.Random,
    *,
    R_max: float,
    z_min: float,
    z_max: float | None = None,
) -> Tuple[float, float, float]:
    if R_max <= 0.0:
        return (0.0, 0.0, max(0.0, z_min))
    R2 = R_max * R_max
    z_lo = max(0.0, float(z_min))
    z_hi = min(R_max, float(z_max)) if z_max is not None else R_max
    if z_hi <= z_lo:
        return (0.0, 0.0, z_hi)
    while True:
        x = rng.uniform(-R_max, R_max)
        y = rng.uniform(-R_max, R_max)
        z = rng.uniform(z_lo, z_hi)
        if x * x + y * y + z * z <= R2:
            return (x, y, z)


def _sample_anchor_mid_hemisphere(
    rng: random.Random,
    *,
    R_max: float,
    z_min: float,
    z_frac_min: float = 0.3,
    z_frac_max: float = 0.6,
) -> Tuple[float, float, float]:
    z_lo = max(z_min, z_frac_min * R_max)
    z_cap = z_frac_max * R_max
    return _sample_anchor_in_hemisphere(rng, R_max=R_max, z_min=z_lo, z_max=z_cap)


def _build_spoke_init_at_origin(
    N: int,
    length: float,
    *,
    phi: float,
    z0: float,
) -> torch.Tensor:
    positions = torch.zeros((N, 3))
    cp, sp = math.cos(phi), math.sin(phi)
    for i in range(N):
        t = length * (i / max(N - 1, 1))
        positions[i, 0] = t * cp
        positions[i, 1] = t * sp
        positions[i, 2] = z0
    return positions


def _apply_init_rotation(
    init_pos: torch.Tensor,
    *,
    rng: random.Random,
    z_shift: float,
    floor_z: float = 0.05,
    max_tilt_deg: float = 20.0,
    allow_tilt: bool,
) -> torch.Tensor:
    p0 = init_pos[0:1]
    rel = init_pos - p0

    phi = rng.uniform(0.0, 2.0 * math.pi)
    theta = 0.0
    if allow_tilt:
        r_max = float(torch.norm(rel, dim=-1).max().item())
        z_min_local = float(rel[..., 2].min().item())
        p0_z = float(init_pos[0, 2].item())
        headroom = p0_z + z_min_local + z_shift - floor_z
        if r_max > 1e-6 and headroom > 0:
            theta_cap = math.asin(min(1.0, headroom / r_max))
        else:
            theta_cap = 0.0
        theta_cap = min(theta_cap, math.radians(max_tilt_deg))
        if theta_cap > 0:
            theta = rng.uniform(-theta_cap, theta_cap)

    cy, sy = math.cos(theta), math.sin(theta)
    cz, sz = math.cos(phi), math.sin(phi)
    Ry = torch.tensor([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=rel.dtype)
    Rz = torch.tensor([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=rel.dtype)
    return p0 + rel @ (Rz @ Ry).T


def build_init_pose(
    *,
    rng: random.Random,
    torch_gen: torch.Generator,
    N: int,
    length: float,
    trajectory_type: str,
    init_state_cfg: Dict[str, Any],
    control_cfg: Dict[str, Any],
    act_cfg: Dict[str, Any],
    random_walk_fraction: float,
    use_anchor: bool,
) -> Tuple[torch.Tensor, int, int, bool]:
    workspace_R = float(control_cfg.get("workspace_radius", 0.9))
    floor_z = float(control_cfg.get("z_min", 0.0))
    anchor_radius = float(init_state_cfg.get("anchor_radius", workspace_R))
    c3d_init_cfg = init_state_cfg.get("circular_3d", {}) or {}
    c3d_anchor_radius_min = float(c3d_init_cfg.get("anchor_radius_min", 0.2))
    floor_margin = float(init_state_cfg.get("floor_margin", 0.05))
    max_tilt_deg = float(init_state_cfg.get("max_tilt_deg", 20.0))
    anchor_z_min = floor_z + max(0.0, floor_margin)
    actuated_init_r = init_state_cfg.get("actuated_init_radius", None)
    if actuated_init_r is not None:
        actuated_init_r = float(actuated_init_r)
    node0_max_r = init_state_cfg.get("node0_max_radius", actuated_init_r)
    if node0_max_r is not None:
        node0_max_r = float(node0_max_r)

    radial_cfg = act_cfg.get("radial_stretch", {}) or {}
    rs_z_frac_min = float(radial_cfg.get("z_frac_min", 0.3))
    rs_z_frac_max = float(radial_cfg.get("z_frac_max", 0.6))

    init_mode = str(init_state_cfg.get("mode", "legacy")).strip().lower()
    spoke_at_origin = init_mode in ("spoke_at_origin", "spoke_hub", "revolve_at_origin")
    hub_z = float(init_state_cfg.get("hub_z", floor_z + floor_margin))
    anchor_at_hub = bool(init_state_cfg.get("anchor_at_hub", spoke_at_origin))

    if spoke_at_origin:
        phi = rng.uniform(0.0, 2.0 * math.pi)
        init_pos = _build_spoke_init_at_origin(N, length, phi=phi, z0=hub_z)
    elif rng.random() < random_walk_fraction:
        init_pos = _random_walk(N, length, torch_gen)
        allow_tilt = False
    else:
        init_pos = generate_horizontal(1, N, length, "cpu")[0].clone()
        init_pos = init_pos - init_pos[0:1].clone()
        allow_tilt = True

    if trajectory_type in ("circular_3d", "radial_stretch"):
        use_anchor = True

    if not spoke_at_origin:
        R_anchor = float(anchor_radius)
        if trajectory_type == "circular_3d":
            R_anchor = max(R_anchor, c3d_anchor_radius_min)

        if trajectory_type == "radial_stretch":
            ax, ay, az = _sample_anchor_mid_hemisphere(
                rng, R_max=workspace_R, z_min=anchor_z_min,
                z_frac_min=rs_z_frac_min, z_frac_max=rs_z_frac_max,
            )
        else:
            ax, ay, az = _sample_anchor_in_hemisphere(rng, R_max=R_anchor, z_min=anchor_z_min)

        init_pos = _apply_init_rotation(
            init_pos, rng=rng, z_shift=az, floor_z=floor_z,
            max_tilt_deg=max_tilt_deg, allow_tilt=allow_tilt,
        )
        init_pos[..., 0] += ax
        init_pos[..., 1] += ay
        init_pos[..., 2] += az

    min_z_sample = float(init_pos[..., 2].min().item())
    if min_z_sample < floor_z:
        init_pos[..., 2] += (floor_z - min_z_sample)

    anchor_idx = 0 if (use_anchor and anchor_at_hub) else (rng.randint(0, N - 1) if use_anchor else -1)
    if use_anchor:
        cand = [j for j in range(1, N)] if anchor_at_hub else [j for j in range(N) if j != anchor_idx]
        cand = cand or [N - 1]
    else:
        cand = list(range(N))
    actuate_idx = (N - 1) if spoke_at_origin else rng.choice(cand)

    if (actuated_init_r is not None and actuated_init_r > 0.0) or (
        node0_max_r is not None and node0_max_r > 0.0
    ):
        for _ in range(6):
            moved = False
            if actuated_init_r is not None and actuated_init_r > 0.0:
                p_act = init_pos[int(actuate_idx)]
                r = float(torch.linalg.norm(p_act).item())
                if r > actuated_init_r + 1e-9:
                    init_pos = init_pos + (p_act * (actuated_init_r / r - 1.0)).unsqueeze(0)
                    moved = True
            if node0_max_r is not None and node0_max_r > 0.0:
                p0 = init_pos[0]
                r0 = float(torch.linalg.norm(p0).item())
                if r0 > node0_max_r + 1e-9:
                    init_pos = init_pos - (p0 - p0 * (node0_max_r / r0)).unsqueeze(0)
                    moved = True
            if not moved:
                break
        min_z_after = float(init_pos[..., 2].min().item())
        if min_z_after < floor_z:
            init_pos[..., 2] += (floor_z - min_z_after)

    return init_pos, anchor_idx, actuate_idx, use_anchor


@torch.no_grad()
def generate_horizontal(B: int, N: int, length: float, device: Union[str, torch.device]) -> torch.Tensor:
    positions = torch.zeros((B, N, 3), device=device)
    x_coords = torch.linspace(0, length, N, device=device)
    positions[..., 0] = x_coords.unsqueeze(0).expand(B, N)
    positions[..., 2] = 0.1
    return positions


@torch.no_grad()
def generate_hanging(
    B: int,
    N: int,
    length: float,
    device: Union[str, torch.device],
    anchor_point: Tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> torch.Tensor:
    positions = torch.zeros((B, N, 3), device=device)
    anchor_tensor = torch.tensor(anchor_point, device=device, dtype=torch.float32)
    z_coords = torch.linspace(
        anchor_tensor[2].item(),
        anchor_tensor[2].item() - length,
        N,
        device=device,
    )
    positions[..., 0] = anchor_tensor[0]
    positions[..., 1] = anchor_tensor[1]
    positions[..., 2] = z_coords.unsqueeze(0).expand(B, N)
    return positions


@torch.no_grad()
def generate_coiled_spiral(B: int, N: int, length: float, device: Union[str, torch.device]) -> torch.Tensor:
    theta = torch.linspace(0, 6 * math.pi, N, device=device)
    x = theta * torch.cos(theta)
    y = theta * torch.sin(theta)
    z = torch.zeros_like(x)
    spiral_points = torch.stack([x, y, z], dim=-1)
    deltas = spiral_points[1:] - spiral_points[:-1]
    scale_factor = length / torch.clamp(torch.sum(torch.norm(deltas, dim=-1)), min=1e-8)
    spiral_points = spiral_points * scale_factor
    return spiral_points.unsqueeze(0).expand(B, N, 3).clone()


@torch.no_grad()
def generate_random_walk(
    B: int,
    N: int,
    length: float,
    device: Union[str, torch.device],
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    positions = torch.zeros((B, N, 3), device=device)
    segment_length = length / (N - 1)
    for i in range(1, N):
        if generator is not None:
            directions = torch.randn((B, 3), device=device, generator=generator)
        else:
            directions = torch.randn((B, 3), device=device)
        directions = directions / torch.clamp(torch.norm(directions, dim=-1, keepdim=True), min=1e-8)
        positions[:, i, :] = positions[:, i - 1, :] + directions * segment_length
    return positions
