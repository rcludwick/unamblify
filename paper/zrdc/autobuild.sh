#!/usr/bin/env bash
set -euo pipefail

# autobuild.sh -- watches unamblify-zrdc.tex and rebuilds unamblify-zrdc.pdf on save

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PATH="/opt/homebrew/bin:$PATH"

if ! command -v fswatch >/dev/null 2>&1; then
    echo "Error: fswatch is required. Install via: brew install fswatch" >&2
    exit 1
fi

echo "============================================================"
echo " Autobuild active: watching unamblify-zrdc.tex"
echo " Output: $SCRIPT_DIR/unamblify-zrdc.pdf"
echo " Press Ctrl+C to stop."
echo "============================================================"

# Initial build
make unamblify-zrdc.pdf || true

fswatch -o unamblify-zrdc.tex bibliography.bib spectrogram_comparison.pdf PXL_20260914_234231998.MACRO_FOCUS.jpg 2>/dev/null | while read -r _; do
    echo ""
    echo "==> Change detected at $(date +%T). Rebuilding unamblify-zrdc.pdf..."
    if make unamblify-zrdc.pdf; then
        echo "==> [OK] Build successful at $(date +%T)"
    else
        echo "==> [FAIL] Build failed at $(date +%T)"
    fi
done
