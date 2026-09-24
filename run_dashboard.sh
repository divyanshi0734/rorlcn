#!/usr/bin/env bash
# ==============================================================================
# Canonical Launcher for RORL Streamlit Dashboard
# Single Canonical Environment: /opt/anaconda3/bin (Python 3.12)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CANONICAL_STREAMLIT="/opt/anaconda3/bin/streamlit"
CANONICAL_PYTHON="/opt/anaconda3/bin/python"

if [ -x "$CANONICAL_STREAMLIT" ]; then
    STREAMLIT_CMD="$CANONICAL_STREAMLIT"
else
    echo "⚠️ Warning: Canonical Streamlit at $CANONICAL_STREAMLIT not found." >&2
    echo "Using system streamlit: $(which streamlit)" >&2
    STREAMLIT_CMD="$(which streamlit)"
fi

echo "======================================================================"
echo "🌐 Route Optimization via Reinforcement Learning (RORL)"
echo "   Canonical Python:    $CANONICAL_PYTHON"
echo "   Streamlit Launcher:  $STREAMLIT_CMD"
echo "   Working Directory:   $SCRIPT_DIR"
echo "======================================================================"

export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"
exec "$STREAMLIT_CMD" run app.py "$@"
