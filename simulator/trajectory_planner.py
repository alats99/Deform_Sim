from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

FORCE_PROFILES = (
    "constant",
    "circular",
    "circular_3d",
    "random_force",
    "radial_stretch",
    "pick_transfer",
)
MAGNITUDE_BINS_DEFAULT = (0.2, 0.45, 0.7, 0.95, 1.2)


def _smoothstep01(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def build_pick_transfer_config(cfg: Dict[str, Any], *, pool_seed: int) -> "PickTransferConfig":
    act = (cfg or {}).get("actuation_ranges", {}) or {}
    pt = act.get("pick_transfer", {}) or {}
    return PickTransferConfig(
        seed=int(pt.get("seed", pool_seed)),
        lift_seconds_min=float(pt.get("lift_seconds_min", 0.5)),
        lift_seconds_max=float(pt.get("lift_seconds_max", 1.0)),
        transfer_seconds_min=float(pt.get("transfer_seconds_min", 0.5)),
        transfer_seconds_max=float(pt.get("transfer_seconds_max", 1.0)),
        lower_seconds=float(pt.get("lower_seconds", 0.3)),
        xy_force_min=float(pt.get("xy_force_min", 0.1)),
        xy_force_max=float(pt.get("xy_force_max", 0.4)),
        xy_force_beta_a=float(pt.get("xy_force_beta_a", 2.0)),
        xy_force_beta_b=float(pt.get("xy_force_beta_b", 6.0)),
        lower_magnitude_min=float(pt.get("lower_magnitude_min", 0.1)),
        lower_magnitude_max=float(pt.get("lower_magnitude_max", 0.2)),
    )


@dataclass
class PickTransferConfig:
    seed: int
    lift_seconds_min: float = 0.5
    lift_seconds_max: float = 1.0
    transfer_seconds_min: float = 0.5
    transfer_seconds_max: float = 1.0
    lower_seconds: float = 0.3
    xy_force_min: float = 0.1
    xy_force_max: float = 0.4
    xy_force_beta_a: float = 2.0
    xy_force_beta_b: float = 6.0
    lower_magnitude_min: float = 0.1
    lower_magnitude_max: float = 0.2

    def pool_meta(self, *, rope_weight_n: float | None = None) -> dict:
        meta = {
            "pick_transfer_seed": int(self.seed),
            "pick_transfer_lift_seconds_min": float(self.lift_seconds_min),
            "pick_transfer_lift_seconds_max": float(self.lift_seconds_max),
            "pick_transfer_transfer_seconds_min": float(self.transfer_seconds_min),
            "pick_transfer_transfer_seconds_max": float(self.transfer_seconds_max),
            "pick_transfer_lower_seconds": float(self.lower_seconds),
            "pick_transfer_xy_force_min": float(self.xy_force_min),
            "pick_transfer_xy_force_max": float(self.xy_force_max),
            "pick_transfer_lower_magnitude_min": float(self.lower_magnitude_min),
            "pick_transfer_lower_magnitude_max": float(self.lower_magnitude_max),
        }
        if rope_weight_n is not None:
            meta["rope_weight_n"] = float(rope_weight_n)
        return meta


class BatchedPickTransfer:
    """Lift (+Fz) → hold and translate (XY) → put-down (−Fz). Forces in Newtons."""

    def __init__(
        self,
        cfg: PickTransferConfig,
        *,
        lift_magnitudes: torch.Tensor,
        lift_steps: torch.Tensor,
        transfer_steps: torch.Tensor,
        dt_outer: float,
        device: torch.device,
    ):
        self.cfg = cfg
        self.device = device
        self.dt_outer = float(dt_outer)
        self.B = int(lift_magnitudes.shape[0])
        self.lift_mag = lift_magnitudes.to(device=device, dtype=torch.float32)
        self.lift_steps = lift_steps.to(device=device, dtype=torch.long).clamp(min=1)
        self.transfer_steps = transfer_steps.to(device=device, dtype=torch.long).clamp(min=1)
        self.lower_steps = max(1, int(round(float(cfg.lower_seconds) / max(dt_outer, 1e-12))))
        self.cycle_steps = self.lift_steps + self.transfer_steps + self.lower_steps
        self._gen = torch.Generator(device="cpu").manual_seed(int(cfg.seed))
        self._fx = torch.zeros((self.B,), dtype=torch.float32, device=device)
        self._fy = torch.zeros((self.B,), dtype=torch.float32, device=device)
        self._lower_mag = torch.zeros((self.B,), dtype=torch.float32, device=device)
        self._cycle_seen = torch.full((self.B,), -1, dtype=torch.long, device=device)
        self._resample_cycle_forces(torch.arange(self.B, device=device))

    def _sample_xy_magnitude(self, n: int) -> torch.Tensor:
        lo = float(self.cfg.xy_force_min)
        hi = float(self.cfg.xy_force_max)
        # Beta-like skew: bias toward lower XY forces during transfer
        skew = float(self.cfg.xy_force_beta_b) / max(float(self.cfg.xy_force_beta_a), 1e-6)
        u = torch.rand((n,), generator=self._gen).pow(skew)
        return lo + (hi - lo) * u

    def _resample_cycle_forces(self, idx: torch.Tensor) -> None:
        if idx.numel() == 0:
            return
        n = int(idx.numel())
        dev = self.device
        lo = float(self.cfg.lower_magnitude_min)
        hi = float(self.cfg.lower_magnitude_max)
        mag_xy = self._sample_xy_magnitude(n).to(dev)
        phi = (torch.rand((n,), generator=self._gen) * (2.0 * math.pi)).to(dev)
        self._fx[idx] = mag_xy * torch.cos(phi)
        self._fy[idx] = mag_xy * torch.sin(phi)
        self._lower_mag[idx] = (lo + (hi - lo) * torch.rand((n,), generator=self._gen)).to(dev)

    @torch.no_grad()
    def get(self, step_in_force: int) -> torch.Tensor:
        step = int(step_in_force)
        step_t = torch.full((self.B,), step, device=self.device, dtype=torch.long)
        cycle_idx = step_t // self.cycle_steps
        phase = step_t % self.cycle_steps

        new_cycle = cycle_idx != self._cycle_seen
        if torch.any(new_cycle):
            idx = torch.nonzero(new_cycle, as_tuple=False).view(-1)
            self._resample_cycle_forces(idx)
            self._cycle_seen = torch.where(new_cycle, cycle_idx, self._cycle_seen)

        out = torch.zeros((self.B, 3), dtype=torch.float32, device=self.device)

        in_lift = phase < self.lift_steps
        xfer_start = self.lift_steps
        xfer_end = self.lift_steps + self.transfer_steps
        in_transfer = (phase >= xfer_start) & (phase < xfer_end)
        in_lower = phase >= xfer_end

        lift_frac = phase.float() / self.lift_steps.float().clamp(min=1.0)
        out[:, 2] = _smoothstep01(lift_frac) * in_lift.float() * self.lift_mag
        out[:, 0] = torch.where(in_transfer, self._fx, out[:, 0])
        out[:, 1] = torch.where(in_transfer, self._fy, out[:, 1])
        out[:, 2] = torch.where(in_transfer, self.lift_mag, out[:, 2])

        lower_local = (phase - xfer_end).clamp(min=0).float()
        lower_denom = torch.full((self.B,), float(self.lower_steps), device=self.device)
        u_lower = _smoothstep01(lower_local / lower_denom) * in_lower.float()
        out[:, 2] = torch.where(in_lower, -u_lower * self._lower_mag, out[:, 2])

        return out

    def meta(self) -> dict:
        return self.cfg.pool_meta()


def profile_id(label: str) -> int:
    return FORCE_PROFILES.index(label)


def normalize_direction(direction_vec: torch.Tensor, *, allow_negative_z: bool = True) -> torch.Tensor:
    out = direction_vec.clone()
    if not allow_negative_z:
        out[..., 2] = torch.abs(out[..., 2])
    return out / torch.clamp(torch.norm(out, dim=-1, keepdim=True), min=1e-8)


def _direction_in_vertical_cone(rng: random.Random, cone_deg: float) -> List[float]:
    cone_rad = math.radians(max(0.0, min(89.0, float(cone_deg))))
    theta = rng.uniform(0.0, cone_rad)
    phi = rng.uniform(0.0, 2.0 * math.pi)
    st = math.sin(theta)
    return [st * math.cos(phi), st * math.sin(phi), math.cos(theta)]


def _random_direction(rng: random.Random, *, upper_hemisphere: bool) -> List[float]:
    while True:
        x, y, z = rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1)
        n = math.sqrt(x * x + y * y + z * z)
        if n > 1e-6:
            break
    if upper_hemisphere:
        z = abs(z)
    return [x / n, y / n, z / n]


def sample_constant_direction(
    rng: random.Random,
    act: Dict[str, Any],
    *,
    allow_neg_fz: bool,
) -> List[float]:
    dir_cfg = act.get("direction", {}) or {}
    cone = dir_cfg.get("vertical_cone_deg")
    if cone is not None:
        return _direction_in_vertical_cone(rng, float(cone))
    return _random_direction(rng, upper_hemisphere=not allow_neg_fz)


def _circular_xy_direction_batched(
    *,
    omegas: torch.Tensor,
    circular_z: torch.Tensor,
    step_in_force: int,
) -> torch.Tensor:
    t = float(step_in_force)
    angle_xy = omegas * t
    circ_dir = torch.stack([torch.cos(angle_xy), torch.sin(angle_xy), circular_z], dim=-1)
    return circ_dir / torch.clamp(circ_dir.norm(dim=-1, keepdim=True), min=1e-8)


def _circular_3d_direction_batched(
    *,
    omegas: torch.Tensor,
    c3d_center: torch.Tensor,
    c3d_z_amp: torch.Tensor,
    c3d_omega_z: torch.Tensor,
    step_in_force: int,
) -> torch.Tensor:
    t = float(step_in_force)
    angle_xy = omegas * t
    z_t = c3d_center[:, 2] + c3d_z_amp * torch.cos(c3d_omega_z * t)
    c3d_dir = torch.stack(
        [c3d_center[:, 0] + torch.cos(angle_xy), c3d_center[:, 1] + torch.sin(angle_xy), z_t],
        dim=-1,
    )
    return c3d_dir / torch.clamp(c3d_dir.norm(dim=-1, keepdim=True), min=1e-8)


def _radial_direction_batched(
    positions: torch.Tensor,
    actuate_idx: torch.Tensor,
    anchor_idx: torch.Tensor,
) -> torch.Tensor:
    b = torch.arange(positions.shape[0], device=positions.device)
    radial = positions[b, actuate_idx] - positions[b, anchor_idx]
    return radial / torch.clamp(radial.norm(dim=-1, keepdim=True), min=1e-8)


@torch.no_grad()
def commanded_force_batched(
    *,
    profile_ids: torch.Tensor,
    const_dirs: torch.Tensor,
    omegas: torch.Tensor,
    magnitudes: torch.Tensor,
    circular_z: torch.Tensor,
    c3d_center: torch.Tensor,
    c3d_z_amp: torch.Tensor,
    c3d_omega_z: torch.Tensor,
    perturb_force: torch.Tensor,
    pick_transfer_force: torch.Tensor,
    step_in_force: int,
    alpha: float,
    positions: torch.Tensor,
    actuate_idx: torch.Tensor,
    anchor_idx: torch.Tensor,
) -> torch.Tensor:
    cmag = (magnitudes * float(alpha)).unsqueeze(-1)

    f_const = const_dirs * cmag
    f_circ = _circular_xy_direction_batched(omegas=omegas, circular_z=circular_z, step_in_force=step_in_force) * cmag
    f_c3d = _circular_3d_direction_batched(
        omegas=omegas, c3d_center=c3d_center, c3d_z_amp=c3d_z_amp,
        c3d_omega_z=c3d_omega_z, step_in_force=step_in_force,
    ) * cmag
    f_rand = perturb_force * float(alpha)
    f_radial = const_dirs * cmag
    f_pick = pick_transfer_force * float(alpha)

    is_const = (profile_ids == profile_id("constant")).unsqueeze(-1)
    is_circ = (profile_ids == profile_id("circular")).unsqueeze(-1)
    is_c3d = (profile_ids == profile_id("circular_3d")).unsqueeze(-1)
    is_rand = (profile_ids == profile_id("random_force")).unsqueeze(-1)
    is_radial = (profile_ids == profile_id("radial_stretch")).unsqueeze(-1)
    is_pick = (profile_ids == profile_id("pick_transfer")).unsqueeze(-1)

    out = torch.zeros_like(f_const)
    out = torch.where(is_const, f_const, out)
    out = torch.where(is_circ, f_circ, out)
    out = torch.where(is_c3d, f_c3d, out)
    out = torch.where(is_rand, f_rand, out)
    out = torch.where(is_radial, f_radial, out)
    out = torch.where(is_pick, f_pick, out)
    return out


class ForceTrajectoryPlanner:
    """Single-sample force planner for visualization and debugging."""

    def __init__(
        self,
        trajectory_type: str = "constant",
        omega: float = 0.002,
        *,
        allow_negative_z: bool = False,
        circular_z_const: float = 0.5,
        circular_3d_z_base: float = 0.0,
        circular_3d_z_amp: float = 1.0,
        circular_3d_omega_z: float | None = None,
        circular_3d_center: tuple = (0.0, 0.0, 0.0),
        random_seed: int,
        device: torch.device = torch.device("cuda"),
        fixed_direction: List[float] | None = None,
    ):
        self.trajectory_type = trajectory_type
        self.omega = float(omega)
        self.allow_negative_z = bool(allow_negative_z)
        self.circular_z_const = float(circular_z_const)
        cx, cy, cz_default = circular_3d_center
        self.circular_3d_cx = float(cx)
        self.circular_3d_cy = float(cy)
        self.circular_3d_cz = float(circular_3d_z_base) if circular_3d_z_base != 0.0 else float(cz_default)
        self.circular_3d_z_amp = float(circular_3d_z_amp)
        self.circular_3d_omega_z = float(circular_3d_omega_z) if circular_3d_omega_z is not None else float(omega)
        self.device = device

        if fixed_direction is not None:
            self.fixed_dir = torch.tensor(fixed_direction, dtype=torch.float32, device=device)
        else:
            gen = torch.Generator(device="cpu").manual_seed(int(random_seed))
            self.fixed_dir = normalize_direction(
                torch.randn(3, generator=gen).to(device),
                allow_negative_z=allow_negative_z,
            )

    def get_direction(self, t: int) -> torch.Tensor:
        if self.trajectory_type in ("constant", "random_force"):
            return self.fixed_dir

        if self.trajectory_type == "circular":
            angle = self.omega * t
            dir_vec = torch.tensor(
                [np.cos(angle), np.sin(angle), self.circular_z_const],
                device=self.device, dtype=torch.float32,
            )
            return normalize_direction(dir_vec, allow_negative_z=self.allow_negative_z)

        if self.trajectory_type == "circular_3d":
            angle = self.omega * t
            z = self.circular_3d_cz + self.circular_3d_z_amp * float(np.cos(self.circular_3d_omega_z * t))
            dir_vec = torch.tensor(
                [self.circular_3d_cx + np.cos(angle), self.circular_3d_cy + np.sin(angle), z],
                device=self.device, dtype=torch.float32,
            )
            return normalize_direction(dir_vec, allow_negative_z=self.allow_negative_z)

        return self.fixed_dir

    def get_force_tensor(
        self,
        t: int,
        B: int,
        N: int,
        actuated_nodes: List[int],
        force_magnitude: float,
    ) -> torch.Tensor:
        direction = self.get_direction(t)
        forces = torch.zeros((B, N, 3), dtype=torch.float32, device=self.device)
        for node_idx in actuated_nodes:
            forces[:, node_idx, :] = force_magnitude * direction
        return forces
