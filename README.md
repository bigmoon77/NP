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

The file starts with the 8-byte magic `NPDS0001`, followed by a uint32 little-endian JSON-header byte length. Captures are sampled at 10, 20, or 30 Hz. The default is a 2-second capture because 100k-node float32 trajectories create large files (about 2.4 MB per frame). The current sphere is the first data-generation source; use different meshes and material settings in later dataset-generation milestones to test cross-mesh generalization.

## Train the first graph-network baseline

Install the Python dependencies:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate
pip install -r training/requirements.txt
```

Put one or more downloaded `.npds` files in `data/`, then run:

```bash
python training/train.py --data data --epochs 100
```

For a meaningful cross-mesh generalization measurement, put trajectories from entirely different held-out shapes in `validation/` and run:

```bash
python training/train.py --data data/train --validation-data data/validation
```

The prototype is a resolution-independent message-passing network: node count is not fixed in the model. It predicts per-vertex acceleration from deformation, velocity, rest position, and mesh adjacency. This is an initial research baseline, not yet a robust general-purpose material model. Current capture UI produces the sphere reference trajectory; collect additional shapes before interpreting validation results as cross-shape generalization.


## Import the public MeshGraphNets dataset

The first external-data experiment uses DeepMind's `deforming_plate` structural-mechanics trajectories. The source files are official TFRecords; the importer converts each trajectory into the NPDS format already used by this project. TensorFlow is not required.

Download the official dataset (all splits):

```powershell
.\.venv\Scripts\python.exe training/download_meshgraphnets.py --dataset deforming_plate --split all
```

Convert a small starter subset (8 trajectories by default):

```powershell
.\.venv\Scripts\python.exe training/convert_meshgraphnets.py --input data/meshgraphnets/deforming_plate --split train --max-trajectories 8 --output data/deforming_plate_train
.\.venv\Scripts\python.exe training/convert_meshgraphnets.py --input data/meshgraphnets/deforming_plate --split valid --max-trajectories 2 --output data/deforming_plate_valid
```

Train and validate:

```powershell
.\.venv\Scripts\python.exe training/train.py --data data/deforming_plate_train --validation-data data/deforming_plate_valid --epochs 100
```

Use `--max-trajectories 0` to convert every trajectory in a split. The converter currently imports fixed-topology triangle meshes and position trajectories. Two-dimensional coordinates are padded with a zero Z coordinate. It derives velocity from consecutive positions and the dataset timestep. The DeformingPlate benchmark is structural mechanics data, not a general 3D soft-body dataset; it is used here to validate the public-data ingestion and training pipeline.

Source: https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets

## Roadmap

1. GPU reference solver and interactive 100k-vertex demo
2. Simulation recording and dataset export
3. GNS-style graph neural network training prototype
4. Learned inference backend and side-by-side accuracy/performance comparison
