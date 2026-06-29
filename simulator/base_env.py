import torch
from typing import Dict, Any, Union


class BaseDeformableEnv:
    def __init__(
        self,
        B: int,
        N_nodes: int,
        device: Union[str, torch.device] = 'cpu',
        **physics_kwargs: Any
    ):
        self.B = B
        self.N_nodes = N_nodes
        self.device = torch.device(device) if isinstance(device, str) else device

        self.positions = torch.zeros((B, N_nodes, 3), device=self.device, dtype=torch.float32)
        self.velocities = torch.zeros((B, N_nodes, 3), device=self.device, dtype=torch.float32)
        self.node_types = torch.zeros((B, N_nodes, 1), device=self.device, dtype=torch.long)

        self.mass_per_node = physics_kwargs.get('mass_per_node', 0.1)
        self.damping = physics_kwargs.get('damping', 5.0)
        self.mu_k = physics_kwargs.get('mu_k', 0.3)
        self.mu_s = physics_kwargs.get('mu_s', 0.5)
        self.restitution = physics_kwargs.get('restitution', 0.0)
        self.dt = physics_kwargs.get('dt', 0.001)

        gravity_vec = physics_kwargs.get('gravity', [0.0, 0.0, -9.81])
        self.gravity = torch.tensor(gravity_vec, device=self.device, dtype=torch.float32)

    def validate_physics(self):
        pass

    @torch.no_grad()
    def reset(self, initial_positions: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.positions = initial_positions.clone().to(self.device).to(torch.float32)
        self.velocities.zero_()
        self.node_types.zero_()
        return self._get_state_dict()

    def _get_state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            'positions': self.positions.clone(),
            'velocities': self.velocities.clone(),
            'node_types': self.node_types.clone()
        }

    def _compute_internal_forces(self) -> torch.Tensor:
        raise NotImplementedError

    def _apply_surface_physics(self, forces: torch.Tensor) -> torch.Tensor:
        contact = (self.positions[..., 2] <= 1e-5) & (forces[..., 2] < 0.0)

        normal_forces_z = torch.zeros_like(forces[..., 2])
        normal_forces_z[contact] = -forces[..., 2][contact]
        forces[..., 2] += normal_forces_z

        v_xy = self.velocities[..., :2]
        v_xy_norm = torch.norm(v_xy, dim=-1)
        f_xy = forces[..., :2]
        f_xy_norm = torch.norm(f_xy, dim=-1)

        is_static = (v_xy_norm < 1e-4) & contact
        is_kinetic = (v_xy_norm >= 1e-4) & contact

        max_static_f = self.mu_s * normal_forces_z
        kinetic_f_mag = self.mu_k * normal_forces_z

        f_xy_dir = f_xy / torch.clamp(f_xy_norm.unsqueeze(-1), min=1e-8)
        v_xy_dir = v_xy / torch.clamp(v_xy_norm.unsqueeze(-1), min=1e-8)

        static_f_applied = torch.clamp(f_xy_norm, max=max_static_f)
        forces[..., :2] = torch.where(
            is_static.unsqueeze(-1),
            f_xy - (static_f_applied.unsqueeze(-1) * f_xy_dir),
            forces[..., :2]
        )

        forces[..., :2] = torch.where(
            is_kinetic.unsqueeze(-1),
            f_xy - (kinetic_f_mag.unsqueeze(-1) * v_xy_dir),
            forces[..., :2]
        )

        return forces

    @torch.no_grad()
    def _step_inplace(self, actuation_forces: torch.Tensor) -> None:
        net_forces = self._compute_internal_forces()
        net_forces += self.mass_per_node * self.gravity.view(1, 1, 3)
        net_forces += actuation_forces

        net_forces = self._apply_surface_physics(net_forces)

        fixed_mask = (self.node_types.squeeze(-1) == 2)
        net_forces[fixed_mask] = 0.0

        accel = net_forces / self.mass_per_node
        self.velocities += accel * self.dt
        self.velocities[fixed_mask] = 0.0

        self.positions += self.velocities * self.dt

        below_floor = self.positions[..., 2] < 0.0
        self.positions[..., 2] = torch.where(below_floor, 0.0, self.positions[..., 2])

        moving_down = self.velocities[..., 2] < 0.0
        self.velocities[..., 2] = torch.where(
            below_floor & moving_down,
            -self.restitution * self.velocities[..., 2],
            self.velocities[..., 2]
        )

    @torch.no_grad()
    def step(self, actuation_forces: torch.Tensor) -> Dict[str, torch.Tensor]:
        self._step_inplace(actuation_forces)
        return self._get_state_dict()

    @torch.no_grad()
    def step_no_return(self, actuation_forces: torch.Tensor) -> None:
        self._step_inplace(actuation_forces)
