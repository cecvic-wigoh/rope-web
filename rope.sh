#!/usr/bin/env bash
# macOS / Linux launcher — equivalent of Rope.bat.
# Usage: ./rope.sh [--config-dir DIR] [--models-dir DIR] [--preset NAME] ...
cd "$(dirname "$0")" || exit 1
if [ -f venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi
exec python Rope.py "$@"
