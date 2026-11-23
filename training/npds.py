from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Trajectory:
    path: Path
    header: dict
    rest_positions: np.ndarray
    edges: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray | None
    node_types: np.ndarray | None
    stress: np.ndarray | None

    @property
    def frame_count(self) -> int:
        return int(self.header["frame_count"])

    @property
    def vertex_count(self) -> int:
        return int(self.header["vertex_count"])

    @property
    def dt(self) -> float:
        return float(self.header.get("capture_dt", 0.0))


def load_npds(path: str | Path) -> Trajectory:
    file_path = Path(path)
    payload = file_path.read_bytes()
    if len(payload) < 12 or payload[:8] not in (b"NPDS0001", b"NPDS0002"):
        raise ValueError(f"{file_path}: invalid NPDS magic or truncated header")

    version = int(payload[7:8].decode("ascii"))
    if version not in (1, 2):
        raise ValueError(f"{file_path}: unsupported NPDS version {version}")
    header_size = struct.unpack_from("<I", payload, 8)[0]
    header_end = 12 + header_size
    if header_end > len(payload):
        raise ValueError(f"{file_path}: truncated JSON header")
    header = json.loads(payload[12:header_end].decode("utf-8"))
    if header.get("format") != "NPDS" or header.get("version") != version:
        raise ValueError(f"{file_path}: header version does not match file magic")

    node_count = int(header["vertex_count"])
    edge_count = int(header["edge_count"])
    frame_count = int(header["frame_count"])
    offset = header_end

    rest_size = node_count * 3 * 4
    edge_size = edge_count * 2 * 4
    node_type_size = node_count * 4 if version == 2 else 0
    frame_value_size = node_count * 3 * 4
    stress_size = node_count * 4 if version == 2 else 0
    per_frame_size = frame_value_size * (2 if version == 1 else 1) + stress_size
    expected_size = offset + rest_size + edge_size + node_type_size + frame_count * per_frame_size
    if len(payload) != expected_size:
        raise ValueError(
            f"{file_path}: byte size mismatch (expected {expected_size}, got {len(payload)})"
        )

    rest = np.frombuffer(payload, dtype="<f4", count=node_count * 3, offset=offset)
    offset += rest_size
    edges_raw = np.frombuffer(payload, dtype="<u4", count=edge_count * 2, offset=offset)
    offset += edge_size

    node_types = None
    if version == 2:
        node_types = np.frombuffer(
            payload, dtype="<i4", count=node_count, offset=offset
        ).astype(np.int64, copy=True)
        offset += node_type_size

    positions = np.empty((frame_count, node_count, 3), dtype=np.float32)
    velocities = np.empty_like(positions) if version == 1 else None
    stress = np.empty((frame_count, node_count), dtype=np.float32) if version == 2 else None

    for frame_index in range(frame_count):
        positions[frame_index] = np.frombuffer(
            payload, dtype="<f4", count=node_count * 3, offset=offset
        ).reshape(node_count, 3)
        offset += frame_value_size
        if version == 1:
            velocities[frame_index] = np.frombuffer(
                payload, dtype="<f4", count=node_count * 3, offset=offset
            ).reshape(node_count, 3)
            offset += frame_value_size
        else:
            stress[frame_index] = np.frombuffer(
                payload, dtype="<f4", count=node_count, offset=offset
            )
            offset += stress_size

    if edge_count and int(edges_raw.max()) >= node_count:
        raise ValueError(f"{file_path}: edge index exceeds vertex count")
    if frame_count < 2:
        raise ValueError(f"{file_path}: at least two frames are required for training")

    return Trajectory(
        path=file_path,
        header=header,
        rest_positions=rest.reshape(node_count, 3).copy(),
        edges=edges_raw.reshape(edge_count, 2).astype(np.int64, copy=True),
        positions=positions,
        velocities=velocities,
        node_types=node_types,
        stress=stress,
    )
