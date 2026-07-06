#!/usr/bin/env python3
"""Alias for fetch_sofascore_passes.py (pass coordinates only)."""

from __future__ import annotations

import sys

print("[onlypasses] iniciando …", flush=True)

try:
    from fetch_sofascore_passes import main
except ImportError as exc:
    print(
        "\nERRO: não encontrou fetch_sofascore_passes.py na pasta scripts/.\n"
        "Você precisa dos arquivos:\n"
        "  scripts/onlypasses.py\n"
        "  scripts/fetch_sofascore_passes.py\n"
        "  scripts/fetch_sofascore_season.py\n"
        "  scripts/sofascore_positions.py\n",
        file=sys.stderr,
        flush=True,
    )
    raise SystemExit(1) from exc

if __name__ == "__main__":
    raise SystemExit(main())
