#!/usr/bin/env bash
# Create .venv and install ore-research in editable mode with search extras.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
if [[ ! -d .venv ]]; then
  "$PY" -m venv .venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
# If PyPI TLS fails on your Mac, uncomment the next two lines:
# python -m pip install certifi
# export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
python -m pip install -e ".[search]"
echo "Done. Run: source .venv/bin/activate   then: ore run 'Your question?' --rounds 4 --delay 12"
echo "Or: .venv/bin/ore run '...'"
