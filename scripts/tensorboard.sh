#!/usr/bin/env bash
# Launch a persistent TensorBoard server on the host pointed at the stratum output directory.
#
# Usage:
#   scripts/tensorboard.sh                    # serves all runs under out/
#   scripts/tensorboard.sh out/my-run         # serves a specific run
#
# Environment:
#   STRATUM_TENSORBOARD_PORT  port to bind (default 6006)
#   STRATUM_TENSORBOARD_HOST  host to bind (default 0.0.0.0)

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

port="${STRATUM_TENSORBOARD_PORT:-6006}"
host="${STRATUM_TENSORBOARD_HOST:-0.0.0.0}"
logdir="${1:-$repo_root/out}"

_ip=$(python3 -c "import socket; s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.connect(('8.8.8.8',80)); print(s.getsockname()[0]); s.close()" 2>/dev/null \
      || hostname -I | awk '{print $1}')

echo "TensorBoard at http://${_ip}:${port}"
echo "logdir: $logdir"
echo

exec python3 -m tensorboard.main \
  --logdir "$logdir" \
  --host "$host" \
  --port "$port"
