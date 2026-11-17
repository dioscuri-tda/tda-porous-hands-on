#!/usr/bin/env bash
# Step 0 - unpack the 100 structures of task B.
#
#   bash 00_prepare_data.sh
#
# data/structures.npz (bit-packed) -> data/structures/sample_000000.raw ... sample_000099.raw
# Each file is a binary 80x80x80 array indexed [z, y, x], 1 = solid, 0 = pore,
# preceded by a 12-byte header (see common/structures.py).
source "$(dirname "$0")/../common/cli.sh"

"$PYTHON" "$COMMON/structures.py" unpack --packed data/structures.npz --out data/structures --format raw
