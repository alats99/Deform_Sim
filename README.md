# Deform_Sim

A batched, GPU-accelerated physics simulator for deformable 1D objects (ropes, cables, strings). Built in PyTorch with no external physics engine dependencies.

The simulator runs B independent environments in parallel using batch tensor operations, which makes it well-suited for generating training data for learning-based robot manipulation systems.

---

## Features

- **Symplectic Euler integration** with configurable timestep
- **Spring-mass rope** with stretch and bending stiffness, dashpot damping, and air drag
- **Capsule self-collision** using segment–segment closest-point detection and a broadphase bounding-sphere filter
- **Ground plane contact** with static and kinetic Coulomb friction and restitution
- **Workspace constraints**: hemisphere reach limit and table-safe force clamping
- **Force profiles**: constant, circular (2D/3D), random perturbation, radial stretch, pick-transfer
- **Procedural initial states**: horizontal, hanging, coiled spiral, random walk, spoke

---

## Requirements

```
torch>=2.0
numpy
```

Install:

```bash
pip install torch numpy
```

---

## Quick Start

```python
import torch
from simulator import BatchedRopeEnv
from simulator.initial_states import generate_horizontal

B = 4       # parallel environments
N = 20      # nodes per rope
length = 0.6

env = BatchedRopeEnv(
    B=B,
    N=N,
    length=length,
    device="cuda",
    stretch_stiffness=2500.0,
    bending_stiffness=100.0,
    damping=0.01,
    drag=0.005,
    mass_per_node=0.008,
    dt=0.001,
)

init_pos = generate_horizontal(B, N, length, device="cuda")
env.reset(init_pos)

# Apply a constant upward force on the last node for 500 steps
for t in range(500):
    forces = torch.zeros((B, N, 3), device="cuda")
    forces[:, -1, 2] = 2.0
    state = env.step(forces)

print(state["positions"].shape)  # (B, N, 3)
```

---

## Project Structure

```
simulator/
  base_env.py              — integrator, gravity, floor contact, friction
  rope_env.py              — spring-mass chain, capsule self-collision
  segment_capsule_collision.py — edge–edge closest-point forces
  collisions.py            — SDF helpers: plane, sphere, AABB
  control.py               — workspace constraints, control stats
  initial_states.py        — procedural initial rope configurations
  perturbation.py          — batched random force perturbations
  trajectory_planner.py    — force profile definitions (constant, circular, etc.)
  sim_config.py            — YAML → physics kwargs helpers
```

---

## Physics Parameters

| Parameter | Default | Description |
|---|---|---|
| `stretch_stiffness` | 2500.0 | Spring stiffness between adjacent nodes [N/m] |
| `bending_stiffness` | 100.0 | Spring stiffness between 2nd-neighbor nodes [N/m] |
| `damping` | 0.01 | Axial dashpot coefficient [N·s/m] |
| `drag` | 0.005 | Per-node air drag [N·s/m] |
| `mass_per_node` | 0.1 | Node mass [kg] |
| `dt` | 0.001 | Timestep [s] |
| `mu_k` | 0.3 | Kinetic friction coefficient |
| `mu_s` | 0.5 | Static friction coefficient |
| `restitution` | 0.0 | Floor bounce coefficient |
| `capsule_self_collision` | True | Enable/disable self-collision |
| `collision_radius` | 0.32 × segment | Capsule half-thickness |

The `validate_physics()` method runs a CFL-like stability check on startup and warns if the timestep is unsafe for the chosen stiffness.

---

## Deployment

### Local

```bash
git clone https://github.com/alats99/Deform_Sim.git
cd Deform_Sim
pip install torch numpy
python -c "from simulator import BatchedRopeEnv; print('OK')"
```

### HPC / SLURM

The simulator has no disk I/O during stepping. A typical SLURM script:

```bash
#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

module load cuda
source activate your_env

python your_datagen_script.py
```

CPU-only mode works by passing `device="cpu"` — no CUDA required.

### Docker

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
WORKDIR /app
COPY . .
RUN pip install numpy
```

---

## Force Profiles

| Profile | Description |
|---|---|
| `constant` | Fixed direction, random or specified |
| `circular` | Rotating XY direction with constant Z component |
| `circular_3d` | Helical: XY rotation + oscillating Z |
| `random_force` | Direction resampled every ~1 second of sim time |
| `radial_stretch` | Pull along anchor → tip axis |
| `pick_transfer` | Lift (+Z) → translate (XY) → lower (−Z) |

---

## TODO

- [ ] **Bézier trajectory planning** — replace piecewise constant force profiles with smooth Bézier curves over the actuation space to produce naturally flowing trajectories
- [ ] **B-spline motion profiles** — extend Bézier support to B-splines for C2 continuity, enabling long-horizon trajectories with guaranteed smoothness
- [ ] **Stable trajectory control** — add a trajectory stabilization module that monitors rope energy / node velocity norm and applies corrective damping or force scaling to prevent runaway oscillations
- [ ] **Cloth topology** — subclass `BaseDeformableEnv` with a 2D grid connectivity (warp + weft + diagonal bending springs) to support cloth simulation alongside ropes
- [ ] **Visualization** — add a minimal viewer (e.g. Open3D or Matplotlib 3D) for real-time inspection of batch rollouts without external tools
