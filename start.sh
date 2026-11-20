#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
exec "${NIGHTSERIES_PYTHON:-python3}" -B app.py
