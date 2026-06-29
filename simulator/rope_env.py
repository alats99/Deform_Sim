import torch
from typing import Dict, Any, Union
from simulator.base_env import BaseDeformableEnv
from simulator.segment_capsule_collision import (
    capsule_self_collision_forces,
    edge_pair_indices,
)


class BatchedRopeEnv(BaseDeformableEnv):
    """
    Batched 1D rope simulator — B independent ropes in parallel.

    Spring topology: stretch (1st-neighbor) + bending (2nd-neighbor) + capsule self-collision.
    Integration: symplectic Euler. Floor contact, gravity, and friction from BaseDeformableEnv.
    """

    def __init__(
        self,
        B: int,
        N: int,
        length: float,
        device: Union[str, torch.device] = 'cpu',
        **physics_kwargs: Any
    ):
        super().__init__(B, N, device, **physics_kwargs)
        self.length = length

        self.stretch_stiffness = physics_kwargs.get('stretch_stiffness', 2500.0)
        self.bending_stiffness = physics_kwargs.get('bending_stiffness', 100.0)

        # Internal spring dashpot [N·s/m] — axial only, no tangential viscosity
        self.damping = physics_kwargs.get('damping', 0.01)
        # Global per-node air drag [N·s/m] — prevents indefinite ringing
        self.drag = physics_kwargs.get('drag', 0.005)

        self.rest_gap_stretch = self.length / (self.N_nodes - 1)
        self.rest_gap_bend = 2.0 * self.rest_gap_stretch

        self.capsule_self_collision = physics_kwargs.get("capsule_self_collision", True)
        default_r = 0.32 * self.rest_gap_stretch
        self.collision_radius = float(physics_kwargs.get("collision_radius", default_r))
        self.collision_stiffness = float(physics_kwargs.get("collision_stiffness", 0.5 * self.stretch_stiffness))
        self.collision_damping = float(physics_kwargs.get("collision_damping", 0.12))
        self.collision_closest_iters = int(physics_kwargs.get("collision_closest_iters", 6))
        ei, ej = edge_pair_indices(self.N_nodes, self.device)
        self._collision_edge_i = ei
        self._collision_edge_j = ej
        self._skip_capsule_collision = False

        self.validate_physics()

    def validate_physics(self):
        import numpy as np
        max_omega = np.sqrt(self.stretch_stiffness / self.mass_per_node)
        stability_limit = 2.0 / max_omega
        if self.dt >= stability_limit:
            print(f"[WARNING] UNSTABLE: dt={self.dt} >= limit={stability_limit:.6f}. "
                  f"Lower dt to ~{stability_limit * 0.8:.6f}.")
        elif self.dt >= stability_limit * 0.5:
            print(f"[NOTE] dt is close to stability limit — watch for jitter.")
        c_crit = 2.0 * np.sqrt(self.stretch_stiffness * self.mass_per_node)
        ratio = self.damping / c_crit
        if ratio > 1.0:
            print(f"[WARNING] OVER-DAMPED: ratio={ratio:.3f}. Rope will behave like wire.")
        elif ratio > 0.7:
            print(f"[NOTE] High damping ratio={ratio:.3f}. Settles fast, looks stiff.")
        else:
            print(f"[OK] Damping ratio={ratio:.3f} — under-damped, realistic rope.")

    def _compute_internal_forces(self) -> torch.Tensor:
        forces = torch.zeros_like(self.positions)

        # Stretch springs: F = k * (|d| - L) * d_hat
        diff_s = self.positions[:, 1:] - self.positions[:, :-1]
        dist_s = torch.norm(diff_s, dim=-1, keepdim=True)
        dir_s = diff_s / torch.clamp(dist_s, min=1e-8)
        f_s = self.stretch_stiffness * (dist_s - self.rest_gap_stretch) * dir_s
        forces[:, :-1] += f_s
        forces[:, 1:] -= f_s

        # Stretch damping: project relative velocity onto spring axis only
        v_diff_s = self.velocities[:, 1:] - self.velocities[:, :-1]
        v_rel_along = (v_diff_s * dir_s).sum(dim=-1, keepdim=True)
        f_damp_s = self.damping * v_rel_along * dir_s
        forces[:, :-1] += f_damp_s
        forces[:, 1:] -= f_damp_s

        # Bending springs: 2nd-neighbor pairs, rest length = 2 * segment length
        if self.N_nodes > 2:
            diff_b = self.positions[:, 2:] - self.positions[:, :-2]
            dist_b = torch.norm(diff_b, dim=-1, keepdim=True)
            dir_b = diff_b / torch.clamp(dist_b, min=1e-8)
            f_b = self.bending_stiffness * (dist_b - self.rest_gap_bend) * dir_b
            forces[:, :-2] += f_b
            forces[:, 2:] -= f_b

            v_diff_b = self.velocities[:, 2:] - self.velocities[:, :-2]
            v_rel_along_b = (v_diff_b * dir_b).sum(dim=-1, keepdim=True)
            bend_damp_coeff = self.damping * (self.bending_stiffness / self.stretch_stiffness)
            f_damp_b = bend_damp_coeff * v_rel_along_b * dir_b
            forces[:, :-2] += f_damp_b
            forces[:, 2:] -= f_damp_b

        # Capsule self-collision between non-adjacent edge pairs
        if (
            self.capsule_self_collision
            and not self._skip_capsule_collision
            and self._collision_edge_i.numel() > 0
            and self.collision_radius > 0.0
            and self.collision_stiffness > 0.0
        ):
            forces += capsule_self_collision_forces(
                self.positions,
                self.velocities,
                self._collision_edge_i,
                self._collision_edge_j,
                self.collision_radius,
                self.collision_stiffness,
                self.collision_damping,
                closest_iters=self.collision_closest_iters,
            )

        # Air drag
        forces -= self.drag * self.velocities

        return forces

    def step(
        self,
        actuation_forces: torch.Tensor,
        *,
        skip_capsule_collision: bool = False,
    ) -> Dict[str, torch.Tensor]:
        prev = self._skip_capsule_collision
        self._skip_capsule_collision = skip_capsule_collision
        try:
            return super().step(actuation_forces)
        finally:
            self._skip_capsule_collision = prev

    @torch.no_grad()
    def step_no_return(
        self,
        actuation_forces: torch.Tensor,
        *,
        skip_capsule_collision: bool = False,
    ) -> None:
        prev = self._skip_capsule_collision
        self._skip_capsule_collision = skip_capsule_collision
        try:
            super().step_no_return(actuation_forces)
        finally:
            self._skip_capsule_collision = prev
