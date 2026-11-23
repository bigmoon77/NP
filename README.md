# NP — Neural Physics Sandbox

A browser-based research sandbox for comparing conventional soft-body simulation with learned physics models.

## Current demo

The first milestone is a GPU-driven, approximately 100k-vertex deformable sphere. Drag the sphere surface with the pointer, release it, and observe its elastic recovery.

> The browser demo currently uses a conventional GPU spring/shape-restoration solver. It is a reference simulator and data-generation baseline, not a trained neural model.

## Run the browser demo

Open `index.html` through a local static HTTP server (WebGL2 is required):

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

## Dataset formats

The browser recorder currently exports NPDS v1. Its binary payload stores rest positions, undirected mesh edges, then per-frame positions and velocities. The file starts with the 8-byte magic `NPDS0001` and a uint32 little-endian JSON-header length.

The MeshGraphNets DeformingPlate importer writes NPDS v2. It stores:

1. Reference mesh positions: float32 [N, 3]
2. Undirected tetrahedral mesh edges: uint32 [E, 2]
3. Node types: int32 [N]
4. Per frame: world positions float32 [N, 3], then stress float32 [N]

NPDS v2 uses the magic `NPDS0002`. The reader supports both versions. V1 remains the browser-capture format; V2 preserves structural-mechanics features that cannot be represented by the old position/velocity-only layout.

## Python setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r training/requirements.txt
```

## DeformingPlate experiment

This experiment uses the public DeepMind MeshGraphNets DeformingPlate dataset. Download the TFRecords (TensorFlow is not required by the converter):

```powershell
.\.venv\Scripts\python.exe training/download_meshgraphnets.py --dataset deforming_plate --split all
```

Convert a starter subset:

```powershell
.\.venv\Scripts\python.exe training/convert_meshgraphnets.py --input data/meshgraphnets/deforming_plate --split train --max-trajectories 8 --output data/deforming_plate_train
.\.venv\Scripts\python.exe training/convert_meshgraphnets.py --input data/meshgraphnets/deforming_plate --split valid --max-trajectories 2 --output data/deforming_plate_valid
```

Train:

```powershell
.\.venv\Scripts\python.exe training/train.py --data data/deforming_plate_train --validation-data data/deforming_plate_valid --epochs 100
```

### Model and boundary conditions

DeformingPlate metadata has `dt=0`; its frames are structural states, not samples at a physical time interval. The importer therefore does not derive physical velocity or acceleration and does not apply a fallback timestep.

The model predicts the next-frame vertex displacement and scalar stress. Node type is encoded as a three-class one-hot feature (raw values 0, 1, and 3). The two constrained classes (1 and 3) are overwritten with their exact next-frame positions during training and evaluation, matching the dataset's prescribed boundary-condition treatment. The network learns the moving vertices (type 0); its displacement loss and rollout metrics are reported on those vertices. Stress is supervised as an auxiliary output.

The trainer uses differentiable autoregressive rollouts, evaluates the complete validation trajectories at the configured interval, and selects the checkpoint by moving-node rollout RMSE. It also reports a zero-displacement baseline, teacher-forced displacement MSE, and stress RMSE. The baseline applies the same exact boundary conditions, so the comparison measures what the learned model adds on unconstrained vertices.

If the converter or NPDS layout changes, regenerate the dataset files before training. In particular, existing NPDS v1 files do not contain the required DeformingPlate node types or stress values.

Source: https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets

## Roadmap

1. GPU reference solver and interactive 100k-vertex demo
2. Simulation recording and dataset export
3. GNS-style graph neural network training prototype
4. Learned inference backend and side-by-side accuracy/performance comparison
