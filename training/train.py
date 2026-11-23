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
    if trajectory.node_types is None or trajectory.stress is None:
        raise ValueError(
            f"{trajectory.path} is NPDS v1. DeformingPlate training requires NPDS v2; "
            "re-run training/convert_meshgraphnets.py."
        )
    if not np.isin(trajectory.node_types, (0, 1, 3)).all():
        raise ValueError(f"{trajectory.path}: unsupported DeformingPlate node types")
    return {
        "source": trajectory,
        "rest": torch.as_tensor(trajectory.rest_positions, dtype=torch.float32, device=device),
        "positions": torch.as_tensor(trajectory.positions, dtype=torch.float32, device=device),
        "stress": torch.as_tensor(trajectory.stress, dtype=torch.float32, device=device),
        "node_types": torch.as_tensor(trajectory.node_types, dtype=torch.long, device=device),
        "edges": make_edge_index(trajectory.edges, device),
    }


def compute_normalization(data_list: list[dict]) -> dict[str, float]:
    edge_squared_sum = 0.0
    edge_count = 0
    displacement_squared_sum = 0.0
    displacement_count = 0
    stress_squared_sum = 0.0
    stress_count = 0

    for data in data_list:
        rest = data["rest"]
        edge_index = data["edges"]
        edge_delta = rest[edge_index[0]] - rest[edge_index[1]]
        edge_squared_sum += torch.sum(edge_delta * edge_delta).item()
        edge_count += edge_delta.shape[0]

        moving_mask = data["node_types"] == 0
        frame_delta = data["positions"][1:] - data["positions"][:-1]
        if torch.any(moving_mask):
            moving_delta = frame_delta[:, moving_mask]
            displacement_squared_sum += torch.sum(moving_delta * moving_delta).item()
            displacement_count += moving_delta.numel()

        stress = data["stress"]
        stress_squared_sum += torch.sum(stress * stress).item()
        stress_count += stress.numel()

    length_scale = float(np.sqrt(edge_squared_sum / max(edge_count, 1)))
    displacement_scale = float(np.sqrt(displacement_squared_sum / max(displacement_count, 1)))
    stress_scale = float(np.sqrt(stress_squared_sum / max(stress_count, 1)))
    return {
        "length_scale": max(length_scale, 1.0e-8),
        "displacement_scale": max(displacement_scale, max(length_scale, 1.0e-8) * 1.0e-6),
        "stress_scale": max(stress_scale, 1.0e-8),
    }


def apply_known_boundary_positions(
    predicted_positions: torch.Tensor,
    exact_next_positions: torch.Tensor,
    node_types: torch.Tensor,
) -> torch.Tensor:
    known_mask = (node_types == 1) | (node_types == 3)
    return torch.where(known_mask.unsqueeze(-1), exact_next_positions, predicted_positions)


def compute_rollout_training_loss(
    model: MeshGraphSimulator,
    data: dict,
    start_frame: int,
    horizon: int,
    normalization: dict[str, float],
) -> torch.Tensor:
    position = data["positions"][start_frame]
    node_types = data["node_types"]
    moving_mask = node_types == 0
    if not torch.any(moving_mask):
        raise ValueError("Trajectory has no moving (node_type=0) vertices")
    step_losses = []

    for step in range(horizon):
        frame = start_frame + step
        target_position = data["positions"][frame + 1]
        target_stress = data["stress"][frame + 1]
        predicted_delta, predicted_stress = model(
            data["rest"], position, node_types, data["edges"], **normalization
        )
        predicted_position = position + predicted_delta
        position = apply_known_boundary_positions(predicted_position, target_position, node_types)

        position_loss = F.mse_loss(
            (position[moving_mask] - target_position[moving_mask]) / normalization["length_scale"],
            torch.zeros_like(target_position[moving_mask]),
        )
        stress_loss = F.mse_loss(
            predicted_stress / normalization["stress_scale"],
            target_stress / normalization["stress_scale"],
        )
        step_losses.append(position_loss + 0.25 * stress_loss)

    return torch.stack(step_losses).mean()


def evaluate_teacher_forced(
    model: MeshGraphSimulator,
    datasets: list[dict],
    normalization: dict[str, float],
) -> tuple[float, float, float]:
    model.eval()
    delta_error_sum = 0.0
    zero_error_sum = 0.0
    stress_error_sum = 0.0
    delta_count = 0
    stress_count = 0

    with torch.no_grad():
        for data in datasets:
            moving_mask = data["node_types"] == 0
            for frame in range(data["source"].frame_count - 1):
                target_delta = data["positions"][frame + 1] - data["positions"][frame]
                predicted_delta, predicted_stress = model(
                    data["rest"], data["positions"][frame], data["node_types"],
                    data["edges"], **normalization,
                )
                difference = predicted_delta[moving_mask] - target_delta[moving_mask]
                delta_error_sum += torch.sum(difference * difference).item()
                zero_error_sum += torch.sum(target_delta[moving_mask] ** 2).item()
                delta_count += target_delta[moving_mask].numel()

                stress_difference = predicted_stress - data["stress"][frame + 1]
                stress_error_sum += torch.sum(stress_difference * stress_difference).item()
                stress_count += stress_difference.numel()

    if delta_count == 0 or stress_count == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        delta_error_sum / delta_count,
        zero_error_sum / delta_count,
        float(np.sqrt(stress_error_sum / stress_count)),
    )


def evaluate_rollout(
    model: MeshGraphSimulator,
    datasets: list[dict],
    normalization: dict[str, float],
) -> dict[str, tuple[float, float]]:
    model.eval()
    names = ("gnn", "zero")
    squared_error_sums = {name: 0.0 for name in names}
    final_squared_error_sums = {name: 0.0 for name in names}
    value_count = 0
    final_value_count = 0

    with torch.no_grad():
        for data in datasets:
            source = data["source"]
            states = {
                name: data["positions"][0].clone()
                for name in names
            }
            node_types = data["node_types"]
            moving_mask = node_types == 0

            for frame in range(source.frame_count - 1):
                target = data["positions"][frame + 1]
                if "gnn" in states:
                    delta, _ = model(
                        data["rest"], states["gnn"], node_types,
                        data["edges"], **normalization,
                    )
                    gnn_position = states["gnn"] + delta
                    states["gnn"] = apply_known_boundary_positions(gnn_position, target, node_types)

                zero_position = states["zero"].clone()
                states["zero"] = apply_known_boundary_positions(zero_position, target, node_types)

                for name in names:
                    difference = states[name][moving_mask] - target[moving_mask]
                    squared_error_sums[name] += torch.sum(difference * difference).item()
                    if frame == source.frame_count - 2:
                        final_squared_error_sums[name] += torch.sum(difference * difference).item()

                value_count += target[moving_mask].numel()
                if frame == source.frame_count - 2:
                    final_value_count += target[moving_mask].numel()

    if value_count == 0 or final_value_count == 0:
        return {name: (float("nan"), float("nan")) for name in names}
    return {
        name: (
            float(np.sqrt(squared_error_sums[name] / value_count)),
            float(np.sqrt(final_squared_error_sums[name] / final_value_count)),
        )
        for name in names
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a resolution-independent quasi-static DeformingPlate graph model."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/deforming_plate_gnn.pt"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=20)
    parser.add_argument("--rollout-horizon", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--message-passing-steps", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluation-interval", type=int, default=10)
    args = parser.parse_args()

    if args.epochs < 1 or args.steps_per_epoch < 1 or args.rollout_horizon < 1:
        raise SystemExit("--epochs, --steps-per-epoch and --rollout-horizon must be positive")
    if args.evaluation_interval < 1:
        raise SystemExit("--evaluation-interval must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but is not available")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )

    train_paths = sorted(args.data.glob("*.npds"))
    if not train_paths:
        raise SystemExit(f"No .npds files found in {args.data}")
    train_trajectories = [load_npds(path) for path in train_paths]

    validation_trajectories = []
    if args.validation_data is not None:
        validation_paths = sorted(args.validation_data.glob("*.npds"))
        if not validation_paths:
            raise SystemExit(f"No validation .npds files found in {args.validation_data}")
        validation_trajectories = [load_npds(path) for path in validation_paths]

    train_data = [prepare_trajectory(item, device) for item in train_trajectories]
    validation_data = [prepare_trajectory(item, device) for item in validation_trajectories]
    evaluation_data = validation_data if validation_data else train_data
    normalization = compute_normalization(train_data)

    print(f"Device: {device}")
    print(f"Training trajectories: {len(train_data)}")
    print("Training meshes: " + ", ".join(f"{item.vertex_count} vertices" for item in train_trajectories))
    print(f"Validation trajectories: {len(validation_data)}")
    print(
        "Normalization: "
        f"length={normalization['length_scale']:.6e}, "
        f"displacement={normalization['displacement_scale']:.6e}, "
        f"stress={normalization['stress_scale']:.6e}"
    )

    model = MeshGraphSimulator(args.hidden_size, args.message_passing_steps).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_loss = float("inf")
    best_epoch = 0
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for _ in range(args.steps_per_epoch):
            data = random.choice(train_data)
            horizon = min(args.rollout_horizon, data["source"].frame_count - 1)
            start_frame = random.randrange(data["source"].frame_count - horizon)
            loss = compute_rollout_training_loss(model, data, start_frame, horizon, normalization)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        should_evaluate = (epoch + 1) % args.evaluation_interval == 0 or epoch + 1 == args.epochs
        if not should_evaluate:
            print(f"epoch {epoch + 1:04d} train={np.mean(losses):.6e} eval=skipped")
            continue

        delta_mse, zero_delta_mse, stress_rmse = evaluate_teacher_forced(
            model, evaluation_data, normalization
        )
        rollout_metrics = evaluate_rollout(model, evaluation_data, normalization)
        rollout_rmse = rollout_metrics["gnn"][0]
        print(
            f"epoch {epoch + 1:04d} train={np.mean(losses):.6e} "
            f"delta_mse={delta_mse:.6e} zero_delta_mse={zero_delta_mse:.6e} "
            f"stress_rmse={stress_rmse:.6e} "
            f"rollout_rmse={rollout_rmse:.6e} "
            f"zero_rollout_rmse={rollout_metrics['zero'][0]:.6e}"
        )
        if rollout_rmse < best_loss:
            best_loss = rollout_rmse
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    final_metrics = evaluate_rollout(model, evaluation_data, normalization)
    delta_mse, zero_delta_mse, stress_rmse = evaluate_teacher_forced(
        model, evaluation_data, normalization
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "model_kind": "deforming_plate_quasistatic",
            "hidden_size": args.hidden_size,
            "message_passing_steps": args.message_passing_steps,
            "input_features": ["position_minus_mesh_pos", "centered_mesh_pos", "node_type_one_hot"],
            "outputs": ["world_pos_delta", "stress"],
            "node_type_values": {"0": "moving", "1": "object", "3": "clamped"},
            "normalization": normalization,
            "best_epoch": best_epoch,
            "best_rollout_rmse": best_loss,
            "rollout_metrics": final_metrics,
            "teacher_forced_delta_mse": delta_mse,
            "zero_delta_baseline_mse": zero_delta_mse,
            "stress_rmse": stress_rmse,
            "seed": args.seed,
        },
        args.output,
    )
    print(f"Best epoch: {best_epoch}")
    print(f"Best moving-node rollout RMSE: {best_loss:.6e}")
    print(f"Teacher-forced displacement MSE: {delta_mse:.6e}")
    print(f"Zero-displacement baseline MSE: {zero_delta_mse:.6e}")
    print(f"Teacher-forced stress RMSE: {stress_rmse:.6e}")
    for name, (all_frames_rmse, final_frame_rmse) in final_metrics.items():
        print(f"Rollout {name}: all_frames_rmse={all_frames_rmse:.6e} final_frame_rmse={final_frame_rmse:.6e}")
    print(f"Saved checkpoint: {args.output}")


if __name__ == "__main__":
    main()
