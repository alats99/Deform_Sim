import torch
from typing import Tuple


@torch.no_grad()
def sdf_plane(
    points: torch.Tensor,
    plane_normal: torch.Tensor,
    plane_point: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    plane_normal = plane_normal.to(dtype=points.dtype, device=points.device)
    plane_normal = plane_normal / torch.norm(plane_normal)
    plane_point = plane_point.to(dtype=points.dtype, device=points.device)
    distances = torch.einsum('bni,i->bn', points - plane_point, plane_normal)
    normals = plane_normal.view(1, 1, 3).expand_as(points)
    return distances, normals


@torch.no_grad()
def sdf_sphere(
    points: torch.Tensor,
    center: torch.Tensor,
    radius: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    center = center.to(dtype=points.dtype, device=points.device)
    p_to_center = points - center
    d = torch.norm(p_to_center, dim=-1)
    distances = d - radius
    normals = p_to_center / torch.clamp(d, min=1e-8).unsqueeze(-1)
    return distances, normals


@torch.no_grad()
def sdf_box(
    points: torch.Tensor,
    center: torch.Tensor,
    extents: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    center = center.to(dtype=points.dtype, device=points.device)
    extents = extents.to(dtype=points.dtype, device=points.device)
    p_local = points - center
    q = torch.abs(p_local) - extents
    q_max = torch.max(q, dim=-1).values
    dist_out = torch.norm(torch.clamp(q, min=0.0), dim=-1)
    dist_in = torch.min(q_max, torch.zeros_like(q_max))
    distances = dist_out + dist_in

    signs = torch.sign(p_local)
    signs = torch.where(signs == 0.0, torch.ones_like(signs), signs)

    normal_out = torch.clamp(q, min=0.0) * signs
    normal_out = normal_out / torch.clamp(torch.norm(normal_out, dim=-1, keepdim=True), min=1e-8)

    q_max_idx = torch.argmax(q, dim=-1)
    normal_in = torch.nn.functional.one_hot(q_max_idx, num_classes=3).to(points.dtype) * signs

    is_outside = (q_max > 0.0).unsqueeze(-1)
    normals = torch.where(is_outside, normal_out, normal_in)

    return distances, normals
