from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class PerturbationConfig:
    seed: int
    mag_min: float = 0.1
    mag_max: float = 1.0
    upper_hemisphere: bool = False
    vertical_cone_deg: float | None = None


class BatchedPerturbation:
    """
    Per-sample random external force, resampled every ~1 second of simulated time.

    Direction: uniform over the full sphere (or upper hemisphere / cone).
    Magnitude: uniform in [mag_min, mag_max].
    """

    def __init__(
        self,
        cfg: PerturbationConfig,
        *,
        B: int,
        dt_outer: float,
        device: torch.device,
    ):
        if cfg.mag_max < cfg.mag_min:
            raise ValueError(f"mag_max ({cfg.mag_max}) < mag_min ({cfg.mag_min})")

        self.cfg = cfg
        self.B = int(B)
        self.dt_outer = float(dt_outer)
        self.device = device
        self.update_period_steps = max(1, int(round(1.0 / max(dt_outer, 1e-12))))

        self._gen = torch.Generator(device="cpu").manual_seed(int(cfg.seed))
        self._direction = torch.zeros((self.B, 3), dtype=torch.float32, device=device)
        self._magnitude = torch.zeros((self.B,), dtype=torch.float32, device=device)
        self._initialized = False

    def _resample(self) -> None:
        if self.cfg.vertical_cone_deg is not None:
            cone_rad = math.radians(max(0.0, min(89.0, float(self.cfg.vertical_cone_deg))))
            theta = torch.rand((self.B,), generator=self._gen) * cone_rad
            phi = torch.rand((self.B,), generator=self._gen) * (2.0 * math.pi)
            st = torch.sin(theta)
            d = torch.stack([st * torch.cos(phi), st * torch.sin(phi), torch.cos(theta)], dim=-1)
        else:
            d = torch.randn((self.B, 3), generator=self._gen)
            if self.cfg.upper_hemisphere:
                d[..., 2] = torch.abs(d[..., 2])
            d = d / torch.clamp(torch.norm(d, dim=-1, keepdim=True), min=1e-8)
        m = self.cfg.mag_min + (self.cfg.mag_max - self.cfg.mag_min) * torch.rand((self.B,), generator=self._gen)
        self._direction = d.to(self.device)
        self._magnitude = m.to(self.device)

    def get(self, t_step: int) -> torch.Tensor:
        if (not self._initialized) or (t_step % self.update_period_steps == 0):
            self._resample()
            self._initialized = True
        return self._direction * self._magnitude.unsqueeze(-1)

    def meta(self) -> dict:
        return {
            "perturb_mag_min": float(self.cfg.mag_min),
            "perturb_mag_max": float(self.cfg.mag_max),
            "perturb_upper_hemisphere": bool(self.cfg.upper_hemisphere),
            "perturb_vertical_cone_deg": self.cfg.vertical_cone_deg,
            "perturb_seed": int(self.cfg.seed),
        }
