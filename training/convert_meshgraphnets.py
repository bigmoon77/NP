from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
from tfrecord import tfrecord_loader


def decode_feature(record: dict, name: str, field: dict) -> np.ndarray:
    if name not in record:
        raise ValueError(f"TFRecord is missing feature '{name}'")
    raw_values = record[name]
    chunks = [bytes(raw_values)] if isinstance(raw_values, (bytes, bytearray, memoryview)) else [bytes(value) for value in raw_values]
    dtype_map = {
        "float32": np.dtype("<f4"),
        "float64": np.dtype("<f8"),
        "int32": np.dtype("<i4"),
        "int64": np.dtype("<i8"),
    }
    dtype_name = str(field["dtype"])
    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported dtype '{dtype_name}' for feature '{name}'")
    values = np.frombuffer(b"".join(chunks), dtype=dtype_map[dtype_name])
    shape = tuple(int(value) for value in field["shape"])
    try:
        values = values.reshape(shape)
    except ValueError as error:
        raise ValueError(f"Cannot reshape feature '{name}' with {values.size} values to {shape}") from error
    if field["type"] == "static":
        if values.ndim >= 1 and values.shape[0] == 1:
            values = values[0]
    elif field["type"] not in ("dynamic", "dynamic_varlen"):
        raise ValueError(f"Unsupported feature type '{field['type']}' for '{name}'")
    return values


def as_xyz(positions: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 3 or positions.shape[-1] not in (2, 3):
        raise ValueError(f"Expected [frames, vertices, 2|3], got {positions.shape}")
    if positions.shape[-1] == 2:
        positions = np.concatenate((positions, np.zeros((*positions.shape[:2], 1), dtype=np.float32)), axis=-1)
    return np.ascontiguousarray(positions, dtype=np.float32)


def as_xyz_static(positions: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[-1] not in (2, 3):
        raise ValueError(f"Expected static [vertices, 2|3], got {positions.shape}")
    if positions.shape[-1] == 2:
        positions = np.concatenate((positions, np.zeros((positions.shape[0], 1), dtype=np.float32)), axis=-1)
    return np.ascontiguousarray(positions, dtype=np.float32)


def cells_to_edges(cells: np.ndarray, vertex_count: int) -> np.ndarray:
    cells = np.asarray(cells, dtype=np.int64)
    if cells.ndim != 2 or cells.shape[1] not in (3, 4):
        raise ValueError(f"Expected triangle/tetrahedron cells [count, 3|4], got {cells.shape}")
    if cells.size == 0 or int(cells.min()) < 0 or int(cells.max()) >= vertex_count:
        raise ValueError("Mesh cells are empty or contain an invalid vertex index")
    local_edges = ((0, 1), (1, 2), (2, 0)) if cells.shape[1] == 3 else (
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)
    )
    edge_parts = np.concatenate([cells[:, edge] for edge in local_edges], axis=0)
    return np.ascontiguousarray(np.unique(np.sort(edge_parts, axis=1), axis=0), dtype="<u4")


def write_deforming_plate_npds(
    destination: Path,
    mesh_positions: np.ndarray,
    edges: np.ndarray,
    node_types: np.ndarray,
    positions: np.ndarray,
    stress: np.ndarray,
    source_dataset: str,
    source_split: str,
    trajectory_index: int,
) -> None:
    frame_count, vertex_count, _ = positions.shape
    if mesh_positions.shape != (vertex_count, 3):
        raise ValueError("mesh_pos and world_pos vertex counts do not match")
    if node_types.shape != (vertex_count,):
        raise ValueError("node_type must contain exactly one value per vertex")
    if stress.shape != (frame_count, vertex_count):
        raise ValueError(f"stress shape {stress.shape} does not match positions")
    if not np.isfinite(mesh_positions).all() or not np.isfinite(positions).all() or not np.isfinite(stress).all():
        raise ValueError("Trajectory contains NaN or infinity")
    if not np.isin(node_types, (0, 1, 3)).all():
        raise ValueError(f"Unexpected DeformingPlate node_type values: {np.unique(node_types)}")

    rest = np.ascontiguousarray(mesh_positions, dtype="<f4")
    edges = np.ascontiguousarray(edges, dtype="<u4")
    node_types = np.ascontiguousarray(node_types, dtype="<i4")
    positions = np.ascontiguousarray(positions, dtype="<f4")
    stress = np.ascontiguousarray(stress, dtype="<f4")
    header = {
        "format": "NPDS",
        "version": 2,
        "dataset_kind": "deforming_plate_quasistatic",
        "coordinate_system": "meshgraphnets_xyz",
        "vertex_count": vertex_count,
        "edge_count": int(edges.shape[0]),
        "frame_count": frame_count,
        "capture_dt": 0.0,
        "source": "DeepMind MeshGraphNets",
        "source_dataset": source_dataset,
        "source_split": source_split,
        "trajectory_index": trajectory_index,
        "node_type_values": {"0": "moving", "1": "object", "3": "clamped"},
        "arrays": [
            "mesh_pos:f32[n,3]",
            "edges:u32[m,2]",
            "node_type:i32[n]",
            "frames:{world_pos:f32[n,3],stress:f32[n]}",
        ],
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        output.write(b"NPDS0002")
        output.write(struct.pack("<I", len(header_bytes)))
        output.write(header_bytes)
        output.write(rest.tobytes())
        output.write(edges.tobytes())
        output.write(node_types.tobytes())
        for frame_index in range(frame_count):
            output.write(positions[frame_index].tobytes())
            output.write(stress[frame_index].tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MeshGraphNets DeformingPlate TFRecords to NPDS v2.")
    parser.add_argument("--input", type=Path, required=True, help="Directory containing meta.json and split TFRecords.")
    parser.add_argument("--split", choices=("train", "valid", "test"), default="train")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-trajectories", type=int, default=8)
    args = parser.parse_args()

    metadata_path = args.input / "meta.json"
    record_path = args.input / f"{args.split}.tfrecord"
    if not metadata_path.is_file() or not record_path.is_file():
        raise SystemExit(f"Expected both {metadata_path} and {record_path}. Run training/download_meshgraphnets.py first.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    fields = metadata["features"]
    required = ("cells", "node_type", "mesh_pos", "world_pos", "stress")
    missing = [name for name in required if name not in fields]
    if missing:
        raise SystemExit(f"DeformingPlate metadata is missing required features: {', '.join(missing)}")
    if float(metadata.get("dt", 0.0)) != 0.0:
        print(f"Metadata dt={metadata.get('dt')}; importing as frame-indexed structural states.")

    description = {name: "byte" for name in metadata["field_names"]}
    output_dir = args.output or Path("data") / f"deforming_plate_{args.split}"
    output_dir.mkdir(parents=True, exist_ok=True)
    loader = tfrecord_loader(str(record_path), None, description)
    converted_count = 0

    for trajectory_index, record in enumerate(loader):
        if args.max_trajectories > 0 and converted_count >= args.max_trajectories:
            break
        mesh_pos = as_xyz_static(decode_feature(record, "mesh_pos", fields["mesh_pos"]))
        node_type = np.asarray(decode_feature(record, "node_type", fields["node_type"]), dtype=np.int64).reshape(-1)
        positions = as_xyz(decode_feature(record, "world_pos", fields["world_pos"]))
        stress = np.asarray(decode_feature(record, "stress", fields["stress"]), dtype=np.float32)
        if stress.ndim == 3 and stress.shape[-1] == 1:
            stress = stress[..., 0]
        if stress.shape != positions.shape[:2]:
            raise ValueError(f"Expected stress [frames, vertices], got {stress.shape}")
        if positions.shape[0] < 2:
            raise ValueError("Trajectory must contain at least two frames")

        cells = decode_feature(record, "cells", fields["cells"])
        if cells.ndim == 3:
            if not np.all(cells == cells[0]):
                raise ValueError("This importer requires fixed topology, but cells change over time")
            cells = cells[0]
        edges = cells_to_edges(cells, positions.shape[1])
        output_path = output_dir / f"trajectory_{trajectory_index:05d}.npds"
        write_deforming_plate_npds(
            output_path, mesh_pos, edges, node_type, positions, stress,
            args.input.name, args.split, trajectory_index,
        )
        converted_count += 1
        print(
            f"[{converted_count}] {output_path}: {positions.shape[1]} vertices, "
            f"{edges.shape[0]} edges, {positions.shape[0]} frames, "
            f"node_types={dict(zip(*np.unique(node_type, return_counts=True)))}"
        )

    if converted_count == 0:
        raise SystemExit("No trajectories were converted")
    print(f"Converted {converted_count} DeformingPlate trajectories into {output_dir}")


if __name__ == "__main__":
    main()
