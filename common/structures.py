r"""
Unpack, load and export the voxelized structures of the tutorial.

The 100 structures of each task are shipped bit-packed in one file,
`data/structures.npz` (about 2-3 MB instead of 51 MB). Every structure is a
binary 80 x 80 x 80 array with 1 = solid and 0 = pore/void.

Two on-disk formats are used, the native ones of the two source repositories:

    .npy   task A (elasticity)    array indexed [x, y, z]
    .raw   task B (permeability)  12-byte header (nx, ny, nz as int32) followed
                                  by one byte per voxel, indexed [z, y, x]

Usage
-----
    python common/structures.py unpack --packed data/structures.npz \
        --out data/structures --format npy|raw
    python common/structures.py to-vtk data/structures/rtp_000.npy rtp_000.vtk

In Python:

    >>> from structures import load_structure
    >>> s = load_structure("data/structures/rtp_000.npy")
    >>> s.shape, 1.0 - s.mean()          # porosity = 1 - solid fraction
"""

import argparse
import os
from pathlib import Path

import numpy as np

RAW_HEADER_BYTES = 12


def save_raw(path, solid):
    """Write a [z, y, x] array of {0, 1} in the .raw format of task B."""
    solid = np.ascontiguousarray(solid, dtype=np.uint8)
    nz, ny, nx = solid.shape
    with open(path, "wb") as fh:
        fh.write(np.array([nx, ny, nz], dtype="<i4").tobytes())
        fh.write(solid.tobytes())


def load_raw(path):
    """Read a .raw structure as uint8, indexed [z, y, x]."""
    raw = Path(path).read_bytes()
    nx, ny, nz = (int(v) for v in np.frombuffer(raw, dtype="<i4", count=3))
    if len(raw) != RAW_HEADER_BYTES + nx * ny * nz:
        raise ValueError(f"{path}: size does not match the header {nx}x{ny}x{nz}")
    return np.frombuffer(raw, dtype=np.uint8, offset=RAW_HEADER_BYTES).reshape(nz, ny, nx)


def load_structure(path):
    """Load a structure (.npy -> [x, y, z], .raw -> [z, y, x]); 1 = solid."""
    path = str(path)
    if path.endswith(".raw"):
        return load_raw(path)
    return np.load(path).astype(np.uint8)


def unpack(packed, out, fmt):
    z = np.load(packed)
    shape = tuple(int(v) for v in z["shape"])
    os.makedirs(out, exist_ok=True)
    for bits, name in zip(z["bits"], z["name"]):
        a = np.unpackbits(bits)[: int(np.prod(shape))].reshape(shape)
        if fmt == "raw":
            save_raw(os.path.join(out, f"{name}.raw"), a)
        else:
            np.save(os.path.join(out, f"{name}.npy"), a)
    print(f"unpacked {len(z['name'])} structures {shape} -> {out}/*.{fmt}")


def to_vtk(src, dst):
    """Legacy binary VTK (STRUCTURED_POINTS) with one point array, `solid`."""
    a = load_structure(src)
    # VTK wants x varying fastest: Fortran order for [x, y, z], C order for [z, y, x]
    if str(src).endswith(".raw"):
        nz, ny, nx = a.shape
        flat = a.ravel(order="C")
    else:
        nx, ny, nz = a.shape
        flat = a.ravel(order="F")
    header = ("# vtk DataFile Version 3.0\n"
              f"{os.path.basename(str(src))}\nBINARY\nDATASET STRUCTURED_POINTS\n"
              f"DIMENSIONS {nx} {ny} {nz}\nORIGIN 0 0 0\nSPACING 1 1 1\n"
              f"POINT_DATA {nx * ny * nz}\nSCALARS solid unsigned_char 1\nLOOKUP_TABLE default\n")
    with open(dst, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(flat.astype(np.uint8).tobytes())
        fh.write(b"\n")
    print(f"wrote {dst}  ({nx}x{ny}x{nz}, solid fraction {flat.mean():.3f})")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("unpack", help="unpack data/structures.npz into one file per structure")
    u.add_argument("--packed", required=True)
    u.add_argument("--out", required=True)
    u.add_argument("--format", choices=("npy", "raw"), required=True)
    v = sub.add_parser("to-vtk", help="export one structure for ParaView")
    v.add_argument("input")
    v.add_argument("output")
    a = p.parse_args()
    if a.cmd == "unpack":
        unpack(a.packed, a.out, a.format)
    else:
        to_vtk(a.input, a.output)


if __name__ == "__main__":
    main()
