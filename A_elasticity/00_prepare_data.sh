#!/usr/bin/env bash
# Step 0 - unpack the 100 structures of task A.
#
#   bash 00_prepare_data.sh
#
# data/structures.npz (bit-packed) -> data/structures/rtp_000.npy ... rtp_099.npy
# Each file is a binary 80x80x80 array indexed [x, y, z], 1 = solid, 0 = void.
source "$(dirname "$0")/../common/cli.sh"

"$PYTHON" "$COMMON/structures.py" unpack --packed data/structures.npz --out data/structures --format npy
