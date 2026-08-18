#!/usr/bin/env python3
"""Collect defensive + duel box-score stats for every match in a SofaScore league.

Lightweight: only ``event`` + ``event_lineups`` per match (~2 API calls each).

Example::

    pip install -r requirements-sofascore.txt
    python -u scripts/fetch_defensive_league.py \\
        --url "https://www.sofascore.com/football/tournament/brazil/brasileirao-serie-a/325#id:72034" \\
        --output ./brasileirao_2025/defensive_stats.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for _path in (SCRIPT_DIR, ROOT):
    _s = str(_path)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from fetch_sofascore_season import list_finished_matches, parse_tournament_url  # noqa: E402

DEFENSIVE_STAT_COLUMNS = [
    "total_tackle",
    "won_tackle",
    "interception_won",
    "total_clearance",
    "outfielder_block",
    "ball_recovery",
    "clearance_off_line",
    "last_man_tackle",
    "error_lead_to_a_shot",
    "error_lead_to_a_goal",
    "challenge_lost",
    "duel_won",
    "duel_lost",
    "aerial_won",
    "aerial_lost",
]

META_COLUMNS = [
    "event_id",
    "match_date",
    "home_team",
    "away_team",
    "player_id",
    "player_name",
    "position",
    "shirt_number",
    "is_home",
    "side",
    "minutes_played",
]

VERBOSE = True


def _log(msg: str = "") -> None:
    if VERBOSE:
        print(msg, flush=True)


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


def fetch_match_defensive_rows(client, event_id: int) -> list[dict]:
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
            stats = entry.statistics
            row = {
                **meta,
                "player_id": entry.player.id,
                "player_name": entry.player.name,
                "position": entry.position_match or entry.player.position or "",
                "shirt_number": entry.shirt_number,
                "is_home": side_name == "home",
                "side": side_name,
                "minutes_played": stats.minutes_played,
            }
            for col in DEFENSIVE_STAT_COLUMNS:
                row[col] = getattr(stats, col, 0)
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch defensive + duel stats for a full SofaScore league season."
    )
    parser.add_argument(
        "--url",
        required=True,
        help="League URL with #id:SEASON fragment",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("defensive_league.csv"),
        help="Output CSV path (parent folder created if needed)",
    )
    parser.add_argument("--rate-limit", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--proxy", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    global VERBOSE
    VERBOSE = not args.quiet

    try:
        from tacoscore import TacosScoreClient
        import pandas as pd
    except ImportError:
        _log("pip install -r requirements-sofascore.txt")
        return 1

    tournament_id, season_id = parse_tournament_url(args.url)
    out_path = args.output.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_path = out_path.with_suffix(".done.json")

    client = TacosScoreClient(
        rate_limit_seconds=args.rate_limit,
        proxies=_resolve_proxies(args.proxy),
    )

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
            rows = fetch_match_defensive_rows(client, eid)
            all_rows.extend(rows)
            done.add(eid)
            done_path.write_text(json.dumps(sorted(done), indent=2), encoding="utf-8")

            tackles = sum(int(r.get("total_tackle", 0) or 0) for r in rows)
            _log(f"  → {len(rows)} jogadores · {tackles} desarmes no jogo")

            pd.DataFrame(all_rows).to_csv(out_path, index=False)
        except Exception as exc:
            _log(f"  ERRO: {type(exc).__name__}: {exc}")
            time.sleep(2.0)

    _log(f"\nConcluído · {len(all_rows)} linhas → {out_path}")
    _log(f"Colunas: {', '.join(META_COLUMNS + DEFENSIVE_STAT_COLUMNS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
