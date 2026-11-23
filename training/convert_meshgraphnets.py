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
    if isinstance(raw_values, (bytes, bytearray, memoryview)):
        chunks = [bytes(raw_values)]
    else:
        chunks = [bytes(value) for value in raw_values]

    dtype_name = str(field["dtype"])
    dtype_map = {
        "float32": np.dtype("<f4"),
        "float64": np.dtype("<f8"),
        "int32": np.dtype("<i4"),
        "int64": np.dtype("<i8"),
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported dtype '{dtype_name}' for feature '{name}'")

    values = np.frombuffer(b"".join(chunks), dtype=dtype_map[dtype_name])
    shape = tuple(int(value) for value in field["shape"])
    try:
        values = values.reshape(shape)
    except ValueError as error:
        raise ValueError(
            f"Cannot reshape feature '{name}' with {values.size} values to {shape}"
        ) from error

    if field["type"] == "static":
        if values.ndim >= 1 and values.shape[0] == 1:
            values = values[0]
    elif field["type"] not in ("dynamic", "dynamic_varlen"):
        raise ValueError(f"Unsupported feature type '{field['type']}' for '{name}'")

    return values


def as_xyz(positions: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 3 or positions.shape[-1] not in (2, 3):
        raise ValueError(
            f"Expected positions shaped [frames, vertices, 2|3], got {positions.shape}"
        )
    if positions.shape[-1] == 2:
        positions = np.concatenate(
            (positions, np.zeros((*positions.shape[:2], 1), dtype=np.float32)),
            axis=-1,
        )
    return np.ascontiguousarray(positions, dtype=np.float32)


def cells_to_edges(cells: np.ndarray, vertex_count: int) -> np.ndarray:
    cells = np.asarray(cells, dtype=np.int64)
    if cells.ndim != 2 or cells.shape[1] not in (3, 4):
        raise ValueError(
            f"Expected triangle or tetrahedron cells [count, 3|4], got {cells.shape}"
        )
    if cells.size == 0:
        raise ValueError("The trajectory contains no mesh cells")
    if int(cells.min()) < 0 or int(cells.max()) >= vertex_count:
        raise ValueError("Mesh cell index is outside the vertex range")

    if cells.shape[1] == 3:
        local_edges = ((0, 1), (1, 2), (2, 0))
    else:
        local_edges = (
            (0, 1), (0, 2), (0, 3),
            (1, 2), (1, 3), (2, 3),
        )

    edge_parts = np.concatenate(
        [cells[:, edge] for edge in local_edges],
        axis=0,
    )
    edges = np.sort(edge_parts, axis=1)
    edges = np.unique(edges, axis=0)
    return np.ascontiguousarray(edges, dtype="<u4")


def write_npds(
    destination: Path,
    positions: np.ndarray,
    edges: np.ndarray,
    dt: float,
    source_dataset: str,
    source_split: str,
    trajectory_index: int,
    timestep_source: str,
) -> None:
    frame_count, vertex_count, _ = positions.shape
    velocities = np.zeros_like(positions, dtype=np.float32)
    if frame_count > 1:
        velocities[1:] = (positions[1:] - positions[:-1]) / dt
        velocities[0] = velocities[1]

    rest_positions = np.ascontiguousarray(positions[0], dtype="<f4")
    positions = np.ascontiguousarray(positions, dtype="<f4")
    velocities = np.ascontiguousarray(velocities, dtype="<f4")
    edges = np.ascontiguousarray(edges, dtype="<u4")

    header = {
        "format": "NPDS",
        "version": 1,
        "coordinate_system": "meshgraphnets_xyz",
        "vertex_count": vertex_count,
        "edge_count": int(edges.shape[0]),
        "frame_count": frame_count,
        "capture_dt": float(dt),
        "timestep_source": timestep_source,
        "source": "DeepMind MeshGraphNets",
        "source_dataset": source_dataset,
        "source_split": source_split,
        "trajectory_index": trajectory_index,
        "arrays": [
            "rest_positions:f32[n,3]",
            "edges:u32[m,2]",
            "frames:{positions:f32[n,3],velocities:f32[n,3]}",
        ],
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        output.write(b"NPDS0001")
        output.write(struct.pack("<I", len(header_bytes)))
        output.write(header_bytes)
        output.write(rest_positions.tobytes())
        output.write(edges.tobytes())
        for frame_index in range(frame_count):
            output.write(positions[frame_index].tobytes())
            output.write(velocities[frame_index].tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert DeepMind MeshGraphNets TFRecords to NPDS trajectories."
    )
    parser.add_argument("--input", type=Path, required=True, help="Dataset directory containing meta.json and a split TFRecord.")
    parser.add_argument("--split", choices=("train", "valid", "test"), default="train")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-trajectories", type=int, default=8)
    parser.add_argument("--position-field", default="world_pos")
    parser.add_argument(
        "--timestep", type=float, default=1.0,
        help="Fallback timestep when metadata has no physical timestep. Defaults to one frame unit.",
    )
    args = parser.parse_args()

    metadata_path = args.input / "meta.json"
    record_path = args.input / f"{args.split}.tfrecord"
    if not metadata_path.is_file() or not record_path.is_file():
        raise SystemExit(
            f"Expected both {metadata_path} and {record_path}. "
            "Run training/download_meshgraphnets.py first."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    fields = metadata["features"]
    if args.position_field not in fields:
        available = ", ".join(fields.keys())
        raise SystemExit(
            f"Position field '{args.position_field}' is absent. Available fields: {available}"
        )
    if "cells" not in fields:
        raise SystemExit("MeshGraphNets metadata has no 'cells' feature")

    metadata_dt = float(metadata.get("dt", 0.0))
    if metadata_dt > 0.0:
        dt = metadata_dt
        timestep_source = "metadata"
    else:
        dt = args.timestep
        timestep_source = "fallback_frame_unit"
        if dt <= 0.0:
            raise SystemExit(f"Fallback timestep must be positive, got {dt}")
        print(
            f"Metadata dt={metadata_dt}; using --timestep={dt} "
            "as a normalized frame interval."
        )

    description = {name: "byte" for name in metadata["field_names"]}
    output_dir = args.output or Path("data") / f"{metadata.get('dataset', 'deforming_plate')}_{args.split}"
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = tfrecord_loader(str(record_path), None, description)
    converted_count = 0
    for trajectory_index, record in enumerate(loader):
        if args.max_trajectories > 0 and converted_count >= args.max_trajectories:
            break

        position_data = decode_feature(record, args.position_field, fields[args.position_field])
        if position_data.ndim != 3:
            raise ValueError(
                f"Position feature must be [frames, vertices, dimensions], got {position_data.shape}"
            )
        positions = as_xyz(position_data)
        if positions.shape[0] < 2:
            raise ValueError("Trajectory must contain at least two frames")

        if not np.isfinite(positions).all():
            raise ValueError("Position trajectory contains NaN or infinity")

        cells = decode_feature(record, "cells", fields["cells"])
        if cells.ndim == 3:
            if not np.all(cells == cells[0]):
                raise ValueError(
                    "This NPDS importer requires fixed topology, but cells change over time"
                )
            cells = cells[0]
        edges = cells_to_edges(cells, positions.shape[1])
        output_path = output_dir / f"trajectory_{trajectory_index:05d}.npds"
        write_npds(
            output_path,
            positions,
            edges,
            dt,
            args.input.name,
            args.split,
            trajectory_index,
            timestep_source,
        )
        converted_count += 1
        print(
            f"[{converted_count}] {output_path}: "
            f"{positions.shape[1]} vertices, {edges.shape[0]} edges, "
            f"{positions.shape[0]} frames"
        )

    if converted_count == 0:
        raise SystemExit("No trajectories were converted")
    print(f"Converted {converted_count} trajectories into {output_dir}")


if __name__ == "__main__":
    main()
