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
    velocities: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.header["frame_count"])

    @property
    def vertex_count(self) -> int:
        return int(self.header["vertex_count"])

    @property
    def dt(self) -> float:
        return float(self.header["capture_dt"])


def load_npds(path: str | Path) -> Trajectory:
    file_path = Path(path)
    payload = file_path.read_bytes()
    if len(payload) < 12 or payload[:8] != b"NPDS0001":
        raise ValueError(f"{file_path}: invalid NPDS magic or truncated header")

    header_size = struct.unpack_from("<I", payload, 8)[0]
    header_end = 12 + header_size
    if header_end > len(payload):
        raise ValueError(f"{file_path}: truncated JSON header")
    header = json.loads(payload[12:header_end].decode("utf-8"))
    if header.get("format") != "NPDS" or header.get("version") != 1:
        raise ValueError(f"{file_path}: unsupported NPDS format")

    node_count = int(header["vertex_count"])
    edge_count = int(header["edge_count"])
    frame_count = int(header["frame_count"])
    offset = header_end

    rest_size = node_count * 3 * 4
    edge_size = edge_count * 2 * 4
    frame_size = node_count * 3 * 4 * 2
    expected_size = offset + rest_size + edge_size + frame_count * frame_size
    if len(payload) != expected_size:
        raise ValueError(
            f"{file_path}: byte size mismatch (expected {expected_size}, got {len(payload)})"
        )

    rest = np.frombuffer(payload, dtype="<f4", count=node_count * 3, offset=offset)
    offset += rest_size
    edges = np.frombuffer(payload, dtype="<u4", count=edge_count * 2, offset=offset)
    offset += edge_size

    positions = np.empty((frame_count, node_count, 3), dtype=np.float32)
    velocities = np.empty_like(positions)
    for frame_index in range(frame_count):
        positions[frame_index] = np.frombuffer(
            payload, dtype="<f4", count=node_count * 3, offset=offset
        ).reshape(node_count, 3)
        offset += node_count * 3 * 4
        velocities[frame_index] = np.frombuffer(
            payload, dtype="<f4", count=node_count * 3, offset=offset
        ).reshape(node_count, 3)
        offset += node_count * 3 * 4

    if edge_count and int(edges.max()) >= node_count:
        raise ValueError(f"{file_path}: edge index exceeds vertex count")
    if frame_count < 2:
        raise ValueError(f"{file_path}: at least two frames are required for training")

    return Trajectory(
        path=file_path,
        header=header,
        rest_positions=rest.reshape(node_count, 3).copy(),
        edges=edges.reshape(edge_count, 2).astype(np.int64, copy=True),
        positions=positions,
        velocities=velocities,
    )
