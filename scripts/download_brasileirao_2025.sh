#!/usr/bin/env bash
# Brasileirão Série A 2025 — download completo via SofaScore
# Uso (na raiz do projeto):
#   bash scripts/download_brasileirao_2025.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

URL="https://www.sofascore.com/football/tournament/brazil/brasileirao-serie-a/325#id:72034"
OUT_DIR="data/sofascore/brasileirao_2025"

echo "Brasileirao 2025 · tournament=325 season=72034"
echo "Saida: $OUT_DIR"
echo

python3 scripts/fetch_sofascore_season.py \
  --url "$URL" \
  --output-dir "$OUT_DIR" \
  --consolidate \
  --resume

echo
echo "Concluido. Arquivos consolidados:"
echo "  $OUT_DIR/season_all.csv"
echo "  $OUT_DIR/player_match_stats.csv"
echo "  $OUT_DIR/season_shots.csv"
