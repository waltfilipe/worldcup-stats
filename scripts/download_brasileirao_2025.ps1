# Brasileirão Série A 2025 — download completo via SofaScore
# Uso (PowerShell, na raiz do projeto):
#   .\scripts\download_brasileirao_2025.ps1

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Url = "https://www.sofascore.com/football/tournament/brazil/brasileirao-serie-a/325#id:72034"
$OutDir = "data\sofascore\brasileirao_2025"

Write-Host "Brasileirao 2025 · tournament=325 season=72034"
Write-Host "Saida: $OutDir"
Write-Host ""

python scripts\fetch_sofascore_season.py `
  --url $Url `
  --output-dir $OutDir `
  --consolidate `
  --resume

if ($LASTEXITCODE -ne 0) {
    Write-Error "Download falhou (exit $LASTEXITCODE). Rode de novo com --resume."
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Concluido. Arquivos consolidados:"
Write-Host "  $OutDir\season_all.csv"
Write-Host "  $OutDir\player_match_stats.csv"
Write-Host "  $OutDir\season_shots.csv"
