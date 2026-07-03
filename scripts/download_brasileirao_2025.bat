@echo off
REM Brasileirao Serie A 2025 — download completo via SofaScore
REM Uso: duplo-clique ou, na raiz do projeto: scripts\download_brasileirao_2025.bat

cd /d "%~dp0\.."

set URL=https://www.sofascore.com/football/tournament/brazil/brasileirao-serie-a/325#id:72034
set OUT_DIR=data\sofascore\brasileirao_2025

echo Brasileirao 2025 · tournament=325 season=72034
echo Saida: %OUT_DIR%
echo.

python scripts\fetch_sofascore_season.py --url "%URL%" --output-dir "%OUT_DIR%" --consolidate --resume

if errorlevel 1 (
  echo Download falhou. Rode de novo para retomar com --resume.
  exit /b 1
)

echo.
echo Concluido. Arquivos consolidados:
echo   %OUT_DIR%\season_all.csv
echo   %OUT_DIR%\player_match_stats.csv
echo   %OUT_DIR%\season_shots.csv
