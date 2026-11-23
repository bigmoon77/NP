# NP — Neural Physics Sandbox

A browser-based research sandbox for comparing conventional soft-body simulation with learned physics models.

## Current demo

The first milestone is a GPU-driven, approximately 100k-vertex deformable sphere. Drag the sphere surface with the pointer, release it, and observe its elastic recovery.

> The initial backend is a conventional GPU spring/shape-restoration solver. It is a reference simulator and data-generation baseline, not a trained neural model. The project is structured so a learned GNS-style backend can be added and compared against it.

## Run

Open `index.html` through a local static HTTP server (WebGL2 is required). For example:

```bash
python -m http.server 8000
```

Then visit http://localhost:8000.

## Controls

- Drag the sphere to pull its surface.
- Use the sliders to tune elasticity, damping, and grab radius.
- Reset restores the initial sphere.
- Orbit with the mouse when not dragging.

## Technical notes

- Three.js + WebGL2
- GPUComputationRenderer ping-pong textures
- 317 × 317 parametric surface grid (100,489 vertices)
- Particle state is stored in floating-point textures
- Spring interactions use the four parametric grid neighbors
- A shape-restoration term helps preserve the spherical rest shape

## Dataset capture

The browser recorder exports `.npds` binary trajectory files. Version 1 stores a JSON metadata header followed by little-endian arrays in this order:

1. Rest positions: float32 [N, 3]
2. Undirected triangle edges: uint32 [E, 2]
3. For each frame: positions float32 [N, 3], then velocities float32 [N, 3]

The file starts with the 8-byte magic `NPDS0001`, followed by a uint32 little-endian JSON-header byte length. Captures are sampled at 10, 20, or 30 Hz. The current sphere is the first data-generation source; use different meshes and material settings in later dataset-generation milestones to test cross-mesh generalization.

## Roadmap

1. GPU reference solver and interactive 100k-vertex demo
2. Simulation recording and dataset export
3. GNS-style graph neural network training prototype
4. Learned inference backend and side-by-side accuracy/performance comparison
