#!/bin/sh
# Build poke-engine for gen9 OU *with terastallization* — the competition config.
#
# CRITICAL FLAGS (a 2026-07-27 regression silently dropped tera for ~a day):
#   --no-default-features : default is gen4, and gen4+gen9 CANNOT coexist
#                           (mutually-exclusive gens -> ~9 duplicate-const errors)
#   --features "gen9 terastallization" : gen9 alone builds a WORKING engine that
#                           silently CANNOT terastallize. tera must be explicit.
# Verify after: engine must offer "-tera" options on untera'd gen9ou positions.
VENV="${VENV:-/home/wiz/Developer/grimoire/crystal-battle/.venv}"
cd "$(dirname "$0")"
VIRTUAL_ENV="$VENV" "$VENV/bin/maturin" develop --release \
  --no-default-features --features "gen9 terastallization" "$@"
