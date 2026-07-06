#!/usr/bin/env python3
"""Atalho para fetch_sofascore_passes.py (passes com coordenadas)."""

from __future__ import annotations

import sys

print("[onlypasses] iniciando …", flush=True)

try:
    from fetch_sofascore_passes import main
except ImportError as exc:
    print(
        "\nERRO: coloque na pasta scripts/:\n"
        "  onlypasses.py\n"
        "  fetch_sofascore_passes.py\n"
        "  sofascore_positions.py\n",
        file=sys.stderr,
        flush=True,
    )
    raise SystemExit(1) from exc

if __name__ == "__main__":
    raise SystemExit(main())
