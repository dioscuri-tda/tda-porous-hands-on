r"""
Direction-aware cone + PCA filtration for the elasticity task (task A).

For a loading direction d, the grid is first re-oriented so that d points along
array axis 2 (an exact axis permutation for [100], [010], [001]; a
nearest-neighbour rotation for the diagonals). Every solid voxel then receives
three filtration values; void voxels receive the sentinel 1.25 in every channel,
so they enter each sublevel set only after all solid voxels.

Channels
--------
0. cone   1 - (solid voxels inside the cone) / (voxels in the cone). The double
          cone has its apex on the voxel and opens along +-d with base radius
          `--cone-radius` at half-height `--cone-height`. 0 = the material
          continues along the loading direction, near 1 = it does not. This
          channel is the one used for persistent homology.
1. dir_y  1 - |v . e_y|   v = leading PCA direction of the solid voxels in a
2. dir_z  1 - |v . e_z|   sphere of radius `--radius` (local strut orientation).

The three channels together are the input of the Euler characteristic profile.

Input: every `<name>.npy` in `--structures` (binary, indexed [x, y, z],
1 = solid). Output: `<outputdir>/<name>.npy`, float32, shape (X, Y, Z, 3).
Existing outputs are skipped.

Usage
-----
    python scripts/cone_filtration.py --structures data/structures \
        --outputdir work/filt/1_0_0 --direction 1 0 0 \
        [--radius 3] [--cone-radius 3] [--cone-height 6] [--workers 8]

Vendored from https://github.com/dioscuri-tda/direction-aware-tda-for-porous-materials
(`scripts/src/filtrations/{speed,pipeline}.py`, merged into one file; the
numerical kernel is unchanged, only the command-line driver is new).
"""

import argparse
import os
from pathlib import Path

import numpy as np
from numba import njit, prange, set_num_threads
from scipy.ndimage import affine_transform
from scipy.spatial.transform import Rotation as ScipyRotation
from tqdm import tqdm

def make_sphere_offsets(radius: int) -> np.ndarray:
    """All (dx,dy,dz) with dx^2+dy^2+dz^2 <= r^2."""
    r2 = radius * radius
    offs = []
    for dx in range(-radius, radius + 1):
        dx2 = dx * dx
        for dy in range(-radius, radius + 1):
            dy2 = dy * dy
            for dz in range(-radius, radius + 1):
                if dx2 + dy2 + dz * dz <= r2:
                    offs.append((dx, dy, dz))
    return np.asarray(offs, dtype=np.int16)


def make_cone_offsets(radius: float, height: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Precompute offsets in the cone bounding box and their weights.
    Weight is 1.0 if inside cone at that dz, else 0.0 (matches Julia get_impact).
    """
    radius = float(radius)
    height = float(abs(height))
    max_x = int(np.ceil(radius))
    max_y = int(np.ceil(radius))
    max_z = int(height)

    offs = []
    ws = []
    if height == 0.0:
        return np.zeros((0, 3), dtype=np.int16), np.zeros((0,), dtype=np.float32)

    for dx in range(-max_x, max_x + 1):
        for dy in range(-max_y, max_y + 1):
            orth = (dx * dx + dy * dy) ** 0.5
            for dz in range(-max_z, max_z + 1):
                z_abs = abs(dz)
                if z_abs > height:
                    w = 0.0
                else:
                    # inside radius proportional to z_abs
                    w = 1.0 if orth <= (radius * z_abs / height) else 0.0
                if w != 0.0:
                    offs.append((dx, dy, dz))
                    ws.append(w)

    return np.asarray(offs, dtype=np.int16), np.asarray(ws, dtype=np.float32)


@njit(inline="always")
def _mod_index(i: int, n: int) -> int:
    return i % n

@njit(inline="always")
def _grid_get(grid: np.ndarray, x: int, y: int, z: int) -> int:
    sx, sy, sz = grid.shape
    return grid[_mod_index(x, sx), _mod_index(y, sy), _mod_index(z, sz)]

@njit
def _power_iteration_top_evec_3x3(C: np.ndarray, iters: int = 12) -> np.ndarray:
    """
    Fast top eigenvector approximation for symmetric 3x3 matrix.
    Good enough for PCA direction; deterministic and numba-friendly.
    """
    v = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    # normalize
    n = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) ** 0.5
    if n != 0.0:
        v /= n

    for _ in range(iters):
        w0 = C[0, 0]*v[0] + C[0, 1]*v[1] + C[0, 2]*v[2]
        w1 = C[1, 0]*v[0] + C[1, 1]*v[1] + C[1, 2]*v[2]
        w2 = C[2, 0]*v[0] + C[2, 1]*v[1] + C[2, 2]*v[2]
        n = (w0*w0 + w1*w1 + w2*w2) ** 0.5
        if n == 0.0:
            break
        v[0], v[1], v[2] = w0/n, w1/n, w2/n
    return v

@njit(inline="always")
def _local_direction_yz_fast(grid: np.ndarray, x: int, y: int, z: int, sphere_offsets: np.ndarray) -> tuple[float, float]:
    """
    Computes PCA direction using running sums (no dataset build).
    Returns (dir_y, dir_z) mapped as 1 - abs(component).
    """
    # Running sums for mean and second moment
    n = 0.0
    sx = sy = sz = 0.0
    sxx = sxy = sxz = 0.0
    syy = syz = 0.0
    szz = 0.0

    for i in range(sphere_offsets.shape[0]):
        dx = int(sphere_offsets[i, 0])
        dy = int(sphere_offsets[i, 1])
        dz = int(sphere_offsets[i, 2])

        if _grid_get(grid, x + dx, y + dy, z + dz) == 0:
            continue

        n += 1.0
        fx = float(dx); fy = float(dy); fz = float(dz)

        sx += fx; sy += fy; sz += fz
        sxx += fx*fx; sxy += fx*fy; sxz += fx*fz
        syy += fy*fy; syz += fy*fz
        szz += fz*fz

    if n < 2.0:
        return 1.0, 1.0

    mx = sx / n
    my = sy / n
    mz = sz / n

    # Covariance = E[vv^T] - m m^T
    C = np.empty((3, 3), dtype=np.float32)
    exx = sxx / n; exy = sxy / n; exz = sxz / n
    eyy = syy / n; eyz = syz / n
    ezz = szz / n

    C[0, 0] = exx - mx*mx
    C[0, 1] = exy - mx*my
    C[0, 2] = exz - mx*mz
    C[1, 0] = C[0, 1]
    C[1, 1] = eyy - my*my
    C[1, 2] = eyz - my*mz
    C[2, 0] = C[0, 2]
    C[2, 1] = C[1, 2]
    C[2, 2] = ezz - mz*mz

    v = _power_iteration_top_evec_3x3(C)

    # Normalize (power iter already normalized, but keep safe)
    nn = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) ** 0.5
    if nn != 0.0:
        v /= nn

    vy = 1.0 - abs(v[1])
    vz = 1.0 - abs(v[2])
    return vy, vz

@njit(parallel=True)
def cone_pca_filtration_fast(
    grid: np.ndarray,
    sphere_offsets: np.ndarray,
    cone_offsets: np.ndarray,
    cone_weights: np.ndarray,
) -> np.ndarray:
    """
    Output: (X,Y,Z,3) => [cone, dir_y, dir_z]
    """
    X, Y, Z = grid.shape
    out = np.empty((X, Y, Z, 3), dtype=np.float32)

    total_cone_w = 0.0
    for i in range(cone_weights.shape[0]):
        total_cone_w += cone_weights[i]

    for x in prange(X):
        for y in range(Y):
            for z in range(Z):
                if grid[x, y, z] == 0:
                    out[x, y, z, 0] = 1.25
                    out[x, y, z, 1] = 1.25
                    out[x, y, z, 2] = 1.25
                    continue

                # Cone channel
                full = 0.0
                for i in range(cone_offsets.shape[0]):
                    dx = int(cone_offsets[i, 0])
                    dy = int(cone_offsets[i, 1])
                    dz = int(cone_offsets[i, 2])
                    if _grid_get(grid, x + dx, y + dy, z + dz) != 0:
                        full += cone_weights[i]

                if total_cone_w == 0.0:
                    cone_val = 1.0
                else:
                    cone_val = 1.0 - (full / total_cone_w)

                out[x, y, z, 0] = cone_val

                # PCA directional channels
                vy, vz = _local_direction_yz_fast(grid, x, y, z, sphere_offsets)
                out[x, y, z, 1] = vy
                out[x, y, z, 2] = vz

    return out


def compute_cone_pca(grid: np.ndarray, radius: int, cone_radius: float, cone_height: float) -> np.ndarray:
    """Filtration of a binary grid along its axis 2. Returns (X, Y, Z, 3) float32."""
    sphere_offsets = make_sphere_offsets(radius)
    cone_offsets, cone_weights = make_cone_offsets(cone_radius, cone_height)
    g = np.ascontiguousarray(grid, dtype=np.uint8)
    return cone_pca_filtration_fast(g, sphere_offsets, cone_offsets, cone_weights)


# ---------------------------------------------------------------------------
# Grid rotation helpers
# ---------------------------------------------------------------------------

# For axis-aligned directions: exact numpy transpose (no interpolation).
# np.transpose(grid, perm) means: new axis i = old axis perm[i].
# We want direction's axis to become axis 2 (z) in the new grid.
_AXIS_PERMS = {
    (1, 0, 0): (1, 2, 0),   # old x → new z
    (0, 1, 0): (0, 2, 1),   # old y → new z
    (0, 0, 1): None,         # already z, no-op
}


def rotate_grid_to_z(grid: np.ndarray, direction: tuple) -> np.ndarray:
    """
    Return a (possibly permuted/rotated) view of `grid` such that
    `direction` aligns with axis 2 (z).  The cone filtration always
    runs along axis 2, so this makes it direction-aware.

    Axis-aligned directions use exact numpy permutation (lossless).
    Diagonal directions use nearest-neighbour scipy rotation (binary grid).
    """
    d = np.array(direction, dtype=float)
    norm = np.linalg.norm(d)
    if norm == 0:
        raise ValueError("Direction vector must be non-zero.")
    d = d / norm

    # Axis-aligned shortcut
    d_int = tuple(int(round(x)) for x in direction)
    if d_int in _AXIS_PERMS:
        perm = _AXIS_PERMS[d_int]
        return np.transpose(grid, perm) if perm is not None else grid

    # General case: rotate so d → z = (0,0,1)
    z = np.array([0.0, 0.0, 1.0])
    cross = np.cross(d, z)
    sin_a = float(np.linalg.norm(cross))
    cos_a = float(np.dot(d, z))

    if sin_a < 1e-10:
        # d already parallel to z (same or opposite direction)
        return grid if cos_a > 0 else grid[:, :, ::-1].copy()

    axis = cross / sin_a
    angle = np.arctan2(sin_a, cos_a)
    R = ScipyRotation.from_rotvec(angle * axis).as_matrix()

    # affine_transform maps output coords → input coords, so use R^T = R^{-1}
    center = (np.array(grid.shape, dtype=float) - 1.0) / 2.0
    offset = center - R.T @ center

    rotated = affine_transform(
        grid.astype(np.float32), R.T,
        offset=offset, order=0, cval=0.0,
        output_shape=grid.shape,
    )
    return (rotated > 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------

def compute(grid: np.ndarray, direction=(0, 0, 1), radius: int = 3,
            cone_radius: float = 3.0, cone_height: float = 6.0) -> np.ndarray:
    """Cone + PCA filtration of `grid` (indexed [x, y, z], 1 = solid) along `direction`."""
    direction = tuple(float(v) for v in direction)
    if direction != (0.0, 0.0, 1.0):
        grid = rotate_grid_to_z(grid, direction)
    return compute_cone_pca(grid, radius, cone_radius, cone_height)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--structures", required=True, help="directory with the .npy structures")
    p.add_argument("--outputdir", required=True)
    p.add_argument("--direction", type=float, nargs=3, default=[0.0, 0.0, 1.0],
                   metavar=("H", "K", "L"), help="loading direction (default: 0 0 1)")
    p.add_argument("--radius", type=int, default=3, help="PCA sphere radius in voxels")
    p.add_argument("--cone-radius", type=float, default=3.0)
    p.add_argument("--cone-height", type=float, default=6.0)
    p.add_argument("--workers", type=int, default=os.cpu_count(), help="numba threads")
    a = p.parse_args()

    set_num_threads(max(1, min(a.workers, os.cpu_count())))
    out = Path(a.outputdir)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.structures).glob("*.npy"))
    if not files:
        raise SystemExit(f"no .npy structures in {a.structures} (run 00_prepare_data.sh first)")

    done = 0
    tag = "[{:g} {:g} {:g}]".format(*a.direction)
    for f in tqdm(files, desc=f"cone filtration {tag}"):
        target = out / f.name
        if target.exists():
            continue
        np.save(target, compute(np.load(f), a.direction, a.radius, a.cone_radius, a.cone_height))
        done += 1
    print(f"  {done} computed, {len(files) - done} cached -> {out}")


if __name__ == "__main__":
    main()
