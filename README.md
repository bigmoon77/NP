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

## Roadmap

1. GPU reference solver and interactive 100k-vertex demo
2. Simulation recording and dataset export
3. GNS-style graph neural network training prototype
4. Learned inference backend and side-by-side accuracy/performance comparison
