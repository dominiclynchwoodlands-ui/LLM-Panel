#!/usr/bin/env bash
# LLM Panel launcher.
# Loads your API keys from a .env file (kept OUTSIDE the code, never committed),
# then starts the MCP server on Bun.
#
# Key handling: no API key is ever stored in the source. They live in .env only.
# Override the .env location with LLM_PANEL_ENV_FILE; otherwise we use the .env
# sitting next to this script.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${LLM_PANEL_ENV_FILE:-$SCRIPT_DIR/.env}"

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

exec bun "$SCRIPT_DIR/server.ts"
