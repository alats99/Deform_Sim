from __future__ import annotations

from typing import Any, Dict


def build_physics_from_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    phys = cfg.get("physics_ranges", {})
    sim = cfg.get("simulator_defaults", {}) or {}

    mass_per_node = phys.get("mass_per_node", None)
    if mass_per_node is None and phys.get("mass_per_meter", None) is not None:
        N = int(sim["N"])
        length = float(sim["length"])
        mass_per_node = float(phys["mass_per_meter"]) * length / float(N)

    physics: Dict[str, Any] = {
        "stretch_stiffness": phys["stretch_stiffness"],
        "bending_stiffness": phys["bending_stiffness"],
        "damping": phys["damping"],
        "drag": phys["drag"],
        "mass_per_node": mass_per_node,
        "gravity": [0.0, 0.0, -9.81],
    }

    col = cfg.get("collision", {}) or {}
    if "enabled" in col and col["enabled"] is not None:
        physics["capsule_self_collision"] = bool(col["enabled"])
    if col.get("stiffness") is not None:
        physics["collision_stiffness"] = float(col["stiffness"])
    elif col.get("stiffness_multiplier") is not None:
        physics["collision_stiffness"] = float(col["stiffness_multiplier"]) * float(phys["stretch_stiffness"])
    if col.get("damping") is not None:
        physics["collision_damping"] = float(col["damping"])
    if col.get("radius") is not None:
        physics["collision_radius"] = float(col["radius"])
    if col.get("closest_iters") is not None:
        physics["collision_closest_iters"] = int(col["closest_iters"])

    return physics


def build_physics_with_bending(cfg: Dict[str, Any], bending_stiffness: float) -> Dict[str, Any]:
    override = dict(cfg)
    phys = dict(cfg.get("physics_ranges", {}) or {})
    phys["bending_stiffness"] = float(bending_stiffness)
    override["physics_ranges"] = phys
    return build_physics_from_cfg(override)


def build_physics_with_mass_scale(cfg: Dict[str, Any], mass_scale: float) -> Dict[str, Any]:
    override = dict(cfg)
    phys = dict(cfg.get("physics_ranges", {}) or {})
    baseline = float(phys.get("mass_per_meter", 0.16))
    phys["mass_per_meter"] = baseline * float(mass_scale)
    phys["mass_per_node"] = None
    override["physics_ranges"] = phys
    return build_physics_from_cfg(override)


def collision_updates_per_dt(cfg: Dict[str, Any], num_substeps: int) -> int:
    col = cfg.get("collision", {}) or {}
    if col.get("updates_per_dt") is not None:
        upd = int(col["updates_per_dt"])
    else:
        every = col.get("every_substep", None)
        upd = num_substeps if (every is None or bool(every)) else 1
    return max(0, min(num_substeps, upd))


def rope_weight_newtons(cfg: Dict[str, Any], *, g: float = 9.81) -> float:
    phys = cfg.get("physics_ranges", {}) or {}
    sim = cfg.get("simulator_defaults", {}) or {}
    mass_per_meter = float(phys.get("mass_per_meter", 0.16))
    length = float(sim.get("length", 0.5))
    return mass_per_meter * length * float(g)
