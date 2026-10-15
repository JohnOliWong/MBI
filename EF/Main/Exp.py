#!/usr/bin/env python3
"""Inspect a local .mat file and print its structure for inspection."""

import argparse
from pathlib import Path


def inspect_value(name: str, value, indent: int = 0) -> None:
    prefix = "  " * indent
    if hasattr(value, "shape"):
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        print(f"{prefix}{name}: ndarray shape={shape}, dtype={dtype}")
    elif isinstance(value, dict):
        print(f"{prefix}{name}: dict with keys={list(value.keys())}")
        for sub_name, sub_value in value.items():
            inspect_value(sub_name, sub_value, indent + 1)
    elif isinstance(value, (list, tuple)):
        print(f"{prefix}{name}: list/tuple len={len(value)}")
        for i, sub_value in enumerate(value[:10]):
            inspect_value(f"{name}[{i}]", sub_value, indent + 1)
        if len(value) > 10:
            print(f"{prefix}  ... ({len(value) - 10} more items)")
    else:
        print(f"{prefix}{name}: {type(value).__name__}")


def inspect_mat(path: Path) -> None:
    print(f"Inspecting MAT file: {path}")
    try:
        from scipy.io import loadmat

        data = loadmat(path, squeeze_me=True, struct_as_record=False)
    except Exception as exc:  # pragma: no cover
        print(f"scipy.io.loadmat failed: {exc}")
        try:
            import h5py

            print("Attempting HDF5 MAT-file read with h5py...")
            with h5py.File(path, "r") as f:
                print("Root groups:", list(f.keys()))
                for key in f.keys():
                    obj = f[key]
                    print(f"  {key}: {type(obj).__name__}, shape={getattr(obj, 'shape', None)}")
            return
        except Exception as exc2:  # pragma: no cover
            raise RuntimeError("Unable to read MAT file with scipy or h5py") from exc2

    filtered = {k: v for k, v in data.items() if not k.startswith("__")}
    if not filtered:
        print("No user variables found in MAT file.")
        return

    print("Top-level variables:")
    for key, value in filtered.items():
        inspect_value(key, value, indent=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a local EOG MAT file.")
    parser.add_argument(
        "--local-file",
        type=str,
        required=True,
        help="Local .mat file path to inspect.",
    )
    args = parser.parse_args()

    mat_path = Path(args.local_file)
    if not mat_path.exists():
        raise FileNotFoundError(f"Local file does not exist: {mat_path}")

    inspect_mat(mat_path)


if __name__ == "__main__":
    main()