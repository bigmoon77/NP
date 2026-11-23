from __future__ import annotations

import argparse
import copy
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
    return {
        "source": trajectory,
        "rest": rest,
        "positions": positions,
        "velocities": velocities,
        "edges": edges,
    }


def evaluate_teacher_forced(
    model: MeshGraphSimulator,
    datasets: list[dict],
) -> tuple[float, float]:
    """Evaluate every consecutive frame pair and a zero-acceleration baseline."""
    model.eval()
    squared_error_sum = 0.0
    baseline_error_sum = 0.0
    value_count = 0

    with torch.no_grad():
        for data in datasets:
            source = data["source"]
            dt = max(source.dt, 1.0e-6)
            for frame in range(source.frame_count - 1):
                target = (
                    data["velocities"][frame + 1] - data["velocities"][frame]
                ) / dt
                predicted = model(
                    data["rest"],
                    data["positions"][frame],
                    data["velocities"][frame],
                    data["edges"],
                )
                squared_error_sum += F.mse_loss(
                    predicted, target, reduction="sum"
                ).item()
                baseline_error_sum += torch.sum(target * target).item()
                value_count += target.numel()

    if value_count == 0:
        return float("nan"), float("nan")
    return squared_error_sum / value_count, baseline_error_sum / value_count


def evaluate_rollout_comparison(
    model: MeshGraphSimulator,
    datasets: list[dict],
) -> dict[str, tuple[float, float]]:
    """Compare GNN, zero-acceleration, and ground-truth-acceleration rollouts."""
    model.eval()
    squared_error_sums = {"gnn": 0.0, "zero": 0.0, "oracle": 0.0}
    final_squared_error_sums = {"gnn": 0.0, "zero": 0.0, "oracle": 0.0}
    value_count = 0
    final_value_count = 0

    with torch.no_grad():
        for data in datasets:
            source = data["source"]
            dt = max(source.dt, 1.0e-6)
            states = {
                name: {
                    "position": data["positions"][0].clone(),
                    "velocity": data["velocities"][0].clone(),
                }
                for name in squared_error_sums
            }

            for frame in range(source.frame_count - 1):
                target_position = data["positions"][frame + 1]
                target_velocity = data["velocities"][frame + 1]
                ground_truth_acceleration = (
                    target_velocity - data["velocities"][frame]
                ) / dt

                accelerations = {
                    "gnn": model(
                        data["rest"],
                        states["gnn"]["position"],
                        states["gnn"]["velocity"],
                        data["edges"],
                    ),
                    "zero": torch.zeros_like(ground_truth_acceleration),
                    "oracle": ground_truth_acceleration,
                }

                for name, acceleration in accelerations.items():
                    state = states[name]
                    state["velocity"] = state["velocity"] + acceleration * dt
                    state["position"] = state["position"] + state["velocity"] * dt
                    difference = state["position"] - target_position
                    squared_error_sums[name] += torch.sum(
                        difference * difference
                    ).item()

                    if frame == source.frame_count - 2:
                        final_squared_error_sums[name] += torch.sum(
                            difference * difference
                        ).item()

                value_count += target_position.numel()
                if frame == source.frame_count - 2:
                    final_value_count += target_position.numel()

    if value_count == 0 or final_value_count == 0:
        return {
            name: (float("nan"), float("nan"))
            for name in squared_error_sums
        }

    return {
        name: (
            float(np.sqrt(squared_error_sums[name] / value_count)),
            float(np.sqrt(final_squared_error_sums[name] / final_value_count)),
        )
        for name in squared_error_sums
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a mesh-resolution-independent graph network."
    )
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument(
        "--output", type=Path, default=Path("checkpoints/mesh_graph_simulator.pt")
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--message-passing-steps", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--validation-data", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--evaluation-interval", type=int, default=10,
        help="Run full-trajectory validation every N epochs (also on the final epoch).",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but is not available")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

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
    print(
        "Training meshes:",
        ", ".join(f"{item.vertex_count} vertices" for item in train_trajectories),
    )
    if validation_data:
        print(f"Validation trajectories: {len(validation_data)}")
    else:
        print("Validation disabled; best checkpoint will be selected on training data.")

    if args.evaluation_interval < 1:
        raise SystemExit("--evaluation-interval must be at least 1")

    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    evaluation_data = validation_data if validation_data else train_data

    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []

        for _ in range(args.steps_per_epoch):
            data = random.choice(train_data)
            source = data["source"]
            frame = random.randrange(source.frame_count - 1)
            dt = max(source.dt, 1.0e-6)

            prediction = model(
                data["rest"],
                data["positions"][frame],
                data["velocities"][frame],
                data["edges"],
            )
            target = (
                data["velocities"][frame + 1] - data["velocities"][frame]
            ) / dt
            loss = F.mse_loss(prediction, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        should_evaluate = (
            (epoch + 1) % args.evaluation_interval == 0
            or epoch + 1 == args.epochs
        )
        if not should_evaluate:
            print(
                f"epoch {epoch + 1:04d} "
                f"train={np.mean(epoch_losses):.6e} eval=skipped"
            )
            continue

        evaluation_loss, baseline_loss = evaluate_teacher_forced(
            model, evaluation_data
        )
        print(
            f"epoch {epoch + 1:04d} "
            f"train={np.mean(epoch_losses):.6e} "
            f"eval={evaluation_loss:.6e} "
            f"zero_baseline={baseline_loss:.6e}"
        )

        if evaluation_loss < best_loss:
            best_loss = evaluation_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")

    model.load_state_dict(best_state)
    rollout_metrics = evaluate_rollout_comparison(model, evaluation_data)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "hidden_size": args.hidden_size,
            "message_passing_steps": args.message_passing_steps,
            "input_features": [
                "position_minus_rest",
                "velocity",
                "rest_position",
            ],
            "output": "acceleration",
            "best_epoch": best_epoch,
            "best_evaluation_mse": best_loss,
            "rollout_metrics": rollout_metrics,
            "seed": args.seed,
        },
        args.output,
    )

    print(f"Best epoch: {best_epoch}")
    print(f"Best evaluation MSE: {best_loss:.6e}")
    print(f"Zero-acceleration baseline MSE: {baseline_loss:.6e}")
    for name, (all_frames_rmse, final_frame_rmse) in rollout_metrics.items():
        print(
            f"Rollout {name}: all_frames_rmse={all_frames_rmse:.6e} "
            f"final_frame_rmse={final_frame_rmse:.6e}"
        )
    print(f"Saved best checkpoint: {args.output}")


if __name__ == "__main__":
    main()
