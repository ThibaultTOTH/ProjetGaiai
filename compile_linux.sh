#!/usr/bin/env bash
# ==============================================================================
#  🚀 Projet Gaia — Script de compilation Linux officiel (Pop!_OS / Ubuntu)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/compile.sh" "$@"
