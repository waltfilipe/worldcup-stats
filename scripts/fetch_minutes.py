#!/usr/bin/env python3
"""Collect player minutes per match from SofaScore.

Lightweight: only ``event`` + ``event_lineups`` per match (~2 API calls each).
Standalone — does not require fetch_sofascore_season.py in the same folder.

Examples::

    python -u fetch_minutes.py \\
        --url "https://www.sofascore.com/football/tournament/spain/laliga/8#id:77559" \\
        --output ./LaLiga_minutes.csv

    python -u fetch_minutes.py --event-id 12813008 --output ./minutes.csv
"""

from __future__ import annotations

print("[minutes] iniciando …", flush=True)

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

OUTPUT_COLUMNS = [
    "event_id",
    "match_date",
    "home_team",
    "away_team",
    "player_id",
    "player_name",
    "position",
    "shirt_number",
    "is_substitute",
    "is_home",
    "side",
    "minutes_played",
]

VERBOSE = True


def _log(msg: str = "") -> None:
    if VERBOSE:
        print(msg, flush=True)


def parse_tournament_url(url: str) -> tuple[int, int]:
    url = url.strip()
    if "#id:" not in url:
        raise ValueError(
            "URL must include #id:SEASON, e.g. .../laliga/8#id:77559"
        )
    path_part, frag = url.split("#id:", 1)
    season_id = int(frag.split("&")[0].split("/")[0].strip())
    path_clean = path_part.rstrip("/").split("?")[0]
    tournament_match = re.search(r"/(\d+)$", path_clean)
    if not tournament_match:
        raise ValueError(f"Could not parse tournament id from: {path_part}")
    return int(tournament_match.group(1)), season_id


def list_finished_matches(client, tournament_id: int, season_id: int):
    _log("  chamando API season_events (pode levar 10–60 s na 1ª vez) …")
    by_round = client.season_events(tournament_id, season_id)
    _log(f"  season_events OK · {len(by_round)} rodadas na resposta")
    matches = []
    seen: set[int] = set()
    for match_list in by_round.values():
        for summary in match_list.events:
            if summary.event_id in seen:
                continue
            seen.add(summary.event_id)
            if summary.is_finished and summary.has_player_statistics:
                matches.append(summary)
    matches.sort(key=lambda m: m.start_timestamp)
    return matches


def _resolve_proxies(proxy_url: str | None) -> dict[str, str] | None:
    url = (
        proxy_url
        or os.environ.get("TACOSCORE_PROXY")
        or os.environ.get("HTTPS_PROXY")
        or os.environ.get("https_proxy")
    )
    if not url:
        return None
    return {"https": url, "http": url}


def fetch_match_minutes_rows(client, event_id: int) -> list[dict]:
    event_detail = client.event(event_id)
    lineups = client.event_lineups(event_id)
    summary = event_detail.summary

    meta = {
        "event_id": event_id,
        "match_date": summary.start_timestamp.isoformat(),
        "home_team": summary.home_team.name,
        "away_team": summary.away_team.name,
    }

    rows: list[dict] = []
    for side_name, side in [("home", lineups.home), ("away", lineups.away)]:
        for entry in side.players:
            rows.append(
                {
                    **meta,
                    "player_id": entry.player.id,
                    "player_name": entry.player.name,
                    "position": entry.position_match or entry.player.position or "",
                    "shirt_number": entry.shirt_number,
                    "is_substitute": entry.is_substitute,
                    "is_home": side_name == "home",
                    "side": side_name,
                    "minutes_played": entry.statistics.minutes_played,
                }
            )
    return rows


def main() -> int:
    _log("[minutes] parseando argumentos …")
    parser = argparse.ArgumentParser(
        description="Fetch player minutes per match from SofaScore."
    )
    parser.add_argument("--url", help="League URL with #id:SEASON")
    parser.add_argument("--event-id", type=int, help="Single match event id")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("minutes_per_match.csv"),
    )
    parser.add_argument("--rate-limit", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--proxy", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not args.url and args.event_id is None:
        parser.error("Provide --url (league) or --event-id (single match)")

    global VERBOSE
    VERBOSE = not args.quiet

    _log("[minutes] importando tacoscore + pandas …")
    try:
        from tacoscore import TacosScoreClient
        import pandas as pd
    except ImportError as exc:
        _log(f"ERRO import: {exc}")
        _log("pip install tacoscore curl_cffi pandas")
        return 1
    _log("[minutes] imports OK")

    out_path = args.output.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _log(f"[minutes] saída → {out_path}")

    client = TacosScoreClient(
        rate_limit_seconds=args.rate_limit,
        proxies=_resolve_proxies(args.proxy),
    )

    if args.event_id is not None:
        eid = args.event_id
        _log(f"Partida única · event_id={eid}")
        rows = fetch_match_minutes_rows(client, eid)
        pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(out_path, index=False)
        total_mins = sum(int(r["minutes_played"] or 0) for r in rows)
        _log(f"→ {len(rows)} jogadores · {total_mins} min totais")
        _log(f"Salvo: {out_path}")
        return 0

    tournament_id, season_id = parse_tournament_url(args.url)
    done_path = out_path.with_suffix(".done.json")

    _log(f"Listando jogos · tournament={tournament_id} season={season_id}")
    matches = list_finished_matches(client, tournament_id, season_id)
    if args.limit:
        matches = matches[: args.limit]
    _log(f"{len(matches)} jogos finalizados com stats de jogador.")

    done: set[int] = set()
    if args.resume and done_path.exists():
        done = set(json.loads(done_path.read_text(encoding="utf-8")))

    all_rows: list[dict] = []
    if args.resume and out_path.exists():
        all_rows = pd.read_csv(out_path).to_dict("records")
        _log(f"Resume: {len(all_rows)} linhas já no CSV.")

    for i, m in enumerate(matches, 1):
        eid = m.event_id
        label = f"{m.home_team.name} {m.display_score} {m.away_team.name}"
        if eid in done:
            _log(f"[{i}/{len(matches)}] SKIP {eid} · {label}")
            continue

        _log(f"[{i}/{len(matches)}] {eid} · {label}")
        try:
            rows = fetch_match_minutes_rows(client, eid)
            all_rows.extend(rows)
            done.add(eid)
            done_path.write_text(json.dumps(sorted(done), indent=2), encoding="utf-8")

            mins = sum(int(r.get("minutes_played", 0) or 0) for r in rows)
            _log(f"  → {len(rows)} jogadores · {mins} min no jogo")

            pd.DataFrame(all_rows, columns=OUTPUT_COLUMNS).to_csv(out_path, index=False)
        except Exception as exc:
            _log(f"  ERRO: {type(exc).__name__}: {exc}")
            time.sleep(2.0)

    _log(f"\nConcluído · {len(all_rows)} linhas → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
