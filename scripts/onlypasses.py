#!/usr/bin/env python3
"""Atalho para fetch_sofascore_passes.py — só coordenadas de passes.

NÃO confundir com fetch_sofascore_season.py (script completo com --categories).

Uso:
    python -u scripts/onlypasses.py --url "..." --output-dir ".\\passbr2026" \\
        --consolidated-only --resume --rate-limit 1.0

Se --help mostrar --categories em vez de --consolidated-only, o arquivo
fetch_sofascore_passes.py na pasta scripts/ está errado — baixe de novo do repo.
"""

from __future__ import annotations

import sys
from pathlib import Path

print("[onlypasses] iniciando …", flush=True)

SCRIPT_DIR = Path(__file__).resolve().parent
_passes_path = SCRIPT_DIR / "fetch_sofascore_passes.py"

if not _passes_path.exists():
    print(
        f"\nERRO: falta {_passes_path.name} na pasta scripts/.\n"
        "Copie do repo:\n"
        "  scripts/onlypasses.py\n"
        "  scripts/fetch_sofascore_passes.py\n"
        "  scripts/sofascore_positions.py\n",
        file=sys.stderr,
        flush=True,
    )
    raise SystemExit(1)

_passes_src = _passes_path.read_text(encoding="utf-8")
if "--consolidated-only" not in _passes_src:
    print(
        f"\nERRO: {_passes_path.name} não tem --consolidated-only.\n"
        "Provavelmente é fetch_sofascore_season.py com nome errado.\n"
        "Baixe fetch_sofascore_passes.py de waltfilipe/worldcup-stats.\n",
        file=sys.stderr,
        flush=True,
    )
    raise SystemExit(1)
if "--categories" in _passes_src and "passes_only" not in _passes_src:
    print(
        f"\nERRO: {_passes_path.name} parece ser o script COMPLETO (--categories).\n"
        "Substitua pelo arquivo de passes do GitHub.\n",
        file=sys.stderr,
        flush=True,
    )
    raise SystemExit(1)

for _path in (SCRIPT_DIR, SCRIPT_DIR.parent):
    _s = str(_path)
    if _s not in sys.path:
        sys.path.insert(0, _s)

try:
    from fetch_sofascore_passes import main
except ImportError as exc:
    print(
        "\nERRO: não foi possível importar fetch_sofascore_passes.\n"
        "Coloque na pasta scripts/:\n"
        "  onlypasses.py\n"
        "  fetch_sofascore_passes.py\n"
        "  sofascore_positions.py\n",
        file=sys.stderr,
        flush=True,
    )
    raise SystemExit(1) from exc

if __name__ == "__main__":
    raise SystemExit(main())
