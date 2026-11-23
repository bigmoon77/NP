from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlopen


BASE_URL = "https://storage.googleapis.com/dm-meshgraphnets"


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Already exists: {destination}")
        return

    temporary_path = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {url}")
    with urlopen(url, timeout=60) as response, temporary_path.open("wb") as output:
        total_size = int(response.headers.get("Content-Length", "0"))
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = downloaded * 100.0 / total_size
                print(f"\\r  {downloaded / 1024 / 1024:.1f} / {total_size / 1024 / 1024:.1f} MiB ({percent:.1f}%)", end="")
        print()

    temporary_path.replace(destination)
    print(f"Saved: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download an official DeepMind MeshGraphNets dataset."
    )
    parser.add_argument("--dataset", default="deforming_plate")
    parser.add_argument("--split", choices=("train", "valid", "test", "all"), default="all")
    parser.add_argument("--output", type=Path, default=Path("data/meshgraphnets"))
    args = parser.parse_args()

    dataset_dir = args.output / args.dataset
    base_url = f"{BASE_URL}/{args.dataset}"
    download_file(f"{base_url}/meta.json", dataset_dir / "meta.json")

    splits = ("train", "valid", "test") if args.split == "all" else (args.split,)
    for split in splits:
        download_file(f"{base_url}/{split}.tfrecord", dataset_dir / f"{split}.tfrecord")


if __name__ == "__main__":
    main()
