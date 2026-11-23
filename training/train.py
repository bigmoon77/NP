from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from npds import Trajectory, load_npds
from model import MeshGraphSimulator


def make_edge_index(edges: np.ndarray, device: torch.device) -> torch.Tensor:
    directed = np.concatenate((edges, edges[:, ::-1]), axis=0)
    return torch.as_tensor(directed.T.copy(), dtype=torch.long, device=device)


def prepare_trajectory(trajectory: Trajectory, device: torch.device) -> dict:
    rest = torch.as_tensor(trajectory.rest_positions, dtype=torch.float32, device=device)
    positions = torch.as_tensor(trajectory.positions, dtype=torch.float32, device=device)
    velocities = torch.as_tensor(trajectory.velocities, dtype=torch.float32, device=device)
    edges = make_edge_index(trajectory.edges, device)
    return {"source": trajectory, "rest": rest, "positions": positions, "velocities": velocities, "edges": edges}


def evaluate(model: MeshGraphSimulator, datasets: list[dict], device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for data in datasets:
            source = data["source"]
            frame = max(0, source.frame_count - 2)
            dt = max(source.dt, 1.0e-6)
            predicted = model(
                data["rest"], data["positions"][frame],
                data["velocities"][frame], data["edges"]
            )
            target = (data["velocities"][frame + 1] - data["velocities"][frame]) / dt
            losses.append(F.mse_loss(predicted, target).item())
    return float(np.mean(losses)) if losses else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a mesh-resolution-independent GNN on NPDS trajectories.")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/mesh_graph_simulator.pt"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--message-passing-steps", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--validation-data", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    train_paths = sorted(args.data.glob("*.npds"))
    if not train_paths:
        raise SystemExit(f"No .npds files found in {args.data}")

    train_trajectories = [load_npds(path) for path in train_paths]
    if args.validation_data:
        validation_paths = sorted(args.validation_data.glob("*.npds"))
        if not validation_paths:
            raise SystemExit(f"No validation .npds files found in {args.validation_data}")
        validation_trajectories = [load_npds(path) for path in validation_paths]
    else:
        validation_trajectories = []

    train_data = [prepare_trajectory(item, device) for item in train_trajectories]
    validation_data = [prepare_trajectory(item, device) for item in validation_trajectories]

    model = MeshGraphSimulator(args.hidden_size, args.message_passing_steps).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    print(f"Device: {device}")
    print(f"Training trajectories: {len(train_data)}")
    print("Training meshes:", ", ".join(f"{item.vertex_count} vertices" for item in train_trajectories))
    if not validation_data:
        print("Validation: disabled. Supply --validation-data with held-out mesh shapes to measure generalization.")

    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        for _ in range(args.steps_per_epoch):
            data = random.choice(train_data)
            source = data["source"]
            frame = random.randrange(source.frame_count - 1)
            dt = max(source.dt, 1.0e-6)
            prediction = model(
                data["rest"], data["positions"][frame],
                data["velocities"][frame], data["edges"]
            )
            target = (data["velocities"][frame + 1] - data["velocities"][frame]) / dt
            loss = F.mse_loss(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        if validation_data:
            validation_loss = evaluate(model, validation_data, device)
            print(f"epoch {epoch + 1:04d} train={np.mean(epoch_losses):.6e} validation={validation_loss:.6e}")
        else:
            print(f"epoch {epoch + 1:04d} train={np.mean(epoch_losses):.6e}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "hidden_size": args.hidden_size,
        "message_passing_steps": args.message_passing_steps,
        "input_features": ["position_minus_rest", "velocity", "rest_position"],
        "output": "acceleration",
    }, args.output)
    print(f"Saved checkpoint: {args.output}")


if __name__ == "__main__":
    main()
