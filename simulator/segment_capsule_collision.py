"""
Capsule self-collision for discrete ropes.

Each rope edge is treated as a thick segment (capsule). Collision is detected via
segment-segment closest-point distance; contact when that distance falls below 2 * radius.
Forces are a linear penalty plus normal damping — no velocity reflections.

Broadphase: bounding-sphere test per pair (no scalar CPU sync).
B==1 path uses fixed (P,) tensors; B>1 path is fully batched.
"""

from __future__ import annotations

import torch


def closest_points_segments_batch(
    p1: torch.Tensor,
    p2: torch.Tensor,
    q1: torch.Tensor,
    q2: torch.Tensor,
    n_iter: int = 6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    d1 = p2 - p1
    d2 = q2 - q1
    a = (d1 * d1).sum(dim=-1).clamp(min=1e-12)
    e = (d2 * d2).sum(dim=-1).clamp(min=1e-12)
    s = torch.full_like(a, 0.5)
    t = torch.full_like(a, 0.5)
    for _ in range(n_iter):
        q_close = q1 + t.unsqueeze(-1) * d2
        s = ((q_close - p1) * d1).sum(dim=-1) / a
        s = s.clamp(0.0, 1.0)
        p_close = p1 + s.unsqueeze(-1) * d1
        t = ((p_close - q1) * d2).sum(dim=-1) / e
        t = t.clamp(0.0, 1.0)
    pc = p1 + s.unsqueeze(-1) * d1
    qc = q1 + t.unsqueeze(-1) * d2
    return pc, qc, s, t


def edge_pair_indices(num_nodes: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    # Edge i connects nodes i and i+1. Non-adjacent pairs share no vertex (j >= i+2).
    n_seg = num_nodes - 1
    ii, jj = torch.meshgrid(
        torch.arange(n_seg, device=device),
        torch.arange(n_seg, device=device),
        indexing="ij",
    )
    m = jj >= ii + 2
    return ii[m], jj[m]


def _capsule_bounding_spheres(
    p1: torch.Tensor,
    p2: torch.Tensor,
    radius: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    c = 0.5 * (p1 + p2)
    half_len = 0.5 * torch.norm(p2 - p1, dim=-1)
    r = half_len + radius
    return c, r


def capsule_self_collision_forces(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    edge_i: torch.Tensor,
    edge_j: torch.Tensor,
    radius: float,
    k_penalty: float,
    k_damping: float,
    closest_iters: int = 6,
) -> torch.Tensor:
    B, N, _ = positions.shape
    P = edge_i.shape[0]

    if B == 1:
        return _capsule_self_collision_forces_b1(
            positions[0], velocities[0],
            edge_i, edge_j,
            radius, k_penalty, k_damping, closest_iters, N,
        ).unsqueeze(0)

    return _capsule_self_collision_forces_general(
        positions, velocities,
        edge_i, edge_j,
        radius, k_penalty, k_damping, closest_iters,
        B, N, P,
    )


def _capsule_self_collision_forces_b1(
    pos: torch.Tensor,
    vel: torch.Tensor,
    edge_i: torch.Tensor,
    edge_j: torch.Tensor,
    radius: float,
    k_penalty: float,
    k_damping: float,
    closest_iters: int,
    N: int,
) -> torch.Tensor:
    p1, p2 = pos[edge_i], pos[edge_i + 1]
    q1, q2 = pos[edge_j], pos[edge_j + 1]
    vp1, vp2 = vel[edge_i], vel[edge_i + 1]
    vq1, vq2 = vel[edge_j], vel[edge_j + 1]

    ci, ri = _capsule_bounding_spheres(p1, p2, radius)
    cj, rj = _capsule_bounding_spheres(q1, q2, radius)
    broad = torch.norm(ci - cj, dim=-1) <= (ri + rj + 1e-7)

    pc, qc, s, t = closest_points_segments_batch(p1, p2, q1, q2, n_iter=closest_iters)
    delta = qc - pc
    dist = torch.norm(delta, dim=-1).clamp(min=1e-8)
    n = delta / dist.unsqueeze(-1)

    penetration = 2.0 * radius - dist
    active = broad & (penetration > 0.0)

    vp = (1.0 - s).unsqueeze(-1) * vp1 + s.unsqueeze(-1) * vp2
    vq = (1.0 - t).unsqueeze(-1) * vq1 + t.unsqueeze(-1) * vq2
    vn = ((vq - vp) * n).sum(dim=-1)

    f_mag = torch.where(active, k_penalty * penetration - k_damping * vn, torch.zeros_like(penetration))
    fm_n = f_mag.unsqueeze(-1) * n

    forces = torch.zeros((N, 3), device=pos.device, dtype=pos.dtype)
    forces.index_add_(0, edge_i, -(1.0 - s).unsqueeze(-1) * fm_n)
    forces.index_add_(0, edge_i + 1, -s.unsqueeze(-1) * fm_n)
    forces.index_add_(0, edge_j, (1.0 - t).unsqueeze(-1) * fm_n)
    forces.index_add_(0, edge_j + 1, t.unsqueeze(-1) * fm_n)

    return forces


def _capsule_self_collision_forces_general(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    edge_i: torch.Tensor,
    edge_j: torch.Tensor,
    radius: float,
    k_penalty: float,
    k_damping: float,
    closest_iters: int,
    B: int,
    N: int,
    P: int,
) -> torch.Tensor:
    device = positions.device

    p1, p2 = positions[:, edge_i], positions[:, edge_i + 1]
    q1, q2 = positions[:, edge_j], positions[:, edge_j + 1]
    vp1, vp2 = velocities[:, edge_i], velocities[:, edge_i + 1]
    vq1, vq2 = velocities[:, edge_j], velocities[:, edge_j + 1]

    ci, ri = _capsule_bounding_spheres(p1, p2, radius)
    cj, rj = _capsule_bounding_spheres(q1, q2, radius)
    broad = torch.norm(ci - cj, dim=-1) <= (ri + rj + 1e-7)

    pc, qc, s, t = closest_points_segments_batch(p1, p2, q1, q2, n_iter=closest_iters)
    delta = qc - pc
    dist = torch.norm(delta, dim=-1).clamp(min=1e-8)
    n = delta / dist.unsqueeze(-1)

    penetration = 2.0 * radius - dist
    active = broad & (penetration > 0.0)

    vp = (1.0 - s).unsqueeze(-1) * vp1 + s.unsqueeze(-1) * vp2
    vq = (1.0 - t).unsqueeze(-1) * vq1 + t.unsqueeze(-1) * vq2
    vn = ((vq - vp) * n).sum(dim=-1)

    f_mag = torch.where(active, k_penalty * penetration - k_damping * vn, torch.zeros_like(penetration))
    fm_n = f_mag.unsqueeze(-1) * n

    forces = torch.zeros_like(positions)
    forces_flat = forces.view(B * N, 3)
    b_idx = torch.arange(B, device=device, dtype=torch.long).view(B, 1).expand(B, P).reshape(-1)

    def scatter(node_idx: torch.Tensor, contrib: torch.Tensor) -> None:
        flat = b_idx * N + node_idx.view(1, P).expand(B, P).reshape(-1)
        forces_flat.index_add_(0, flat, contrib.reshape(B * P, 3))

    scatter(edge_i, -(1.0 - s).unsqueeze(-1) * fm_n)
    scatter(edge_i + 1, -s.unsqueeze(-1) * fm_n)
    scatter(edge_j, (1.0 - t).unsqueeze(-1) * fm_n)
    scatter(edge_j + 1, t.unsqueeze(-1) * fm_n)

    return forces
