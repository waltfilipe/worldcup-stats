#!/usr/bin/env python3
"""Download **pass coordinates only** from SofaScore (lightweight harvest).

Uses far fewer API calls than ``fetch_sofascore_season.py``:

- **This script:** ``lineups`` + one ``rating-breakdown`` per player who played
- **Full script:** event, stats, incidents, graph, shotmap, heatmap, etc.

Typical league match: ~23 requests vs ~50–80 with the full pipeline.

Output is compatible with ``app.py`` (``season_all.csv`` schema, passes only).

Example::

    pip install -r requirements-sofascore.txt
    python scripts/fetch_sofascore_passes.py \\
        --url "https://www.sofascore.com/football/tournament/italy/serie-a/23#id:XXXXX" \\
        --consolidated-only --resume --rate-limit 2.5 --copy-season-to-root
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for _path in (SCRIPT_DIR, ROOT):
    _s = str(_path)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from fetch_sofascore_season import (  # noqa: E402
    ACTION_COLUMNS,
    _proxy_log_label,
    _resolve_proxies,
    load_done,
    parse_tournament_url,
    save_done,
)
from sofascore_positions import resolve_match_positions  # noqa: E402

DEFAULT_OUT = ROOT / "data" / "sofascore"
PASS_ACTION_TYPES = frozenset({"pass", "cross", "throw-in"})


def _log(message: str = "") -> None:
    """Print immediately (no buffering) — important on Windows PowerShell."""
    print(message, flush=True)


def list_finished_matches_verbose(client, tournament_id: int, season_id: int):
    """Like list_finished_matches but prints each round as it is fetched."""
    rounds = client.tournament_rounds(tournament_id, season_id)
    total_rounds = len(rounds.rounds)
    _log(f"  {total_rounds} rodadas no calendário — consultando SofaScore …")

    matches = []
    seen: set[int] = set()
    finished_in_round = 0

    for idx, round_info in enumerate(rounds.rounds, start=1):
        round_label = round_info.slug or f"rodada {round_info.round}"
        _log(f"  · [{idx}/{total_rounds}] {round_label} …")
        match_list = client.round_events(
            tournament_id, season_id, round_info.round, round_info.slug
        )
        round_finished = 0
        for summary in match_list.events:
            if summary.event_id in seen:
                continue
            seen.add(summary.event_id)
            if summary.is_finished and summary.has_player_statistics:
                matches.append(summary)
                round_finished += 1
                _log(
                    f"      + {summary.event_id}  {summary.start_timestamp:%Y-%m-%d}  "
                    f"{summary.home_team.name} {summary.display_score} {summary.away_team.name}"
                )
        finished_in_round += round_finished
        _log(
            f"    → {len(match_list.events)} jogos na rodada · "
            f"{round_finished} finalizados com stats (total acumulado: {finished_in_round})"
        )

    matches.sort(key=lambda m: m.start_timestamp)
    return matches


def _median(values: list[float]) -> float:
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def fetch_match_passes(
    client,
    event_id: int,
    match_summary,
    *,
    min_minutes: int = 1,
    verbose: bool = True,
) -> tuple[list[dict], int, int]:
    """Fetch pass rows for one match using lineups + rating-breakdown only."""
    from tacoscore.exceptions import NotFoundError

    if verbose:
        _log("  · baixando escalação …")
    lineups = client.event_lineups(event_id)
    meta = {
        "event_id": event_id,
        "home_team": match_summary.home_team.name,
        "away_team": match_summary.away_team.name,
        "match_date": match_summary.start_timestamp.isoformat(),
    }

    raw_position_by_id: dict[int, str] = {}
    for entry in lineups.all_players():
        raw_position_by_id[entry.player.id] = (
            entry.position_match or entry.player.position or ""
        )

    eligible = [
        entry
        for entry in lineups.all_players()
        if entry.statistics.minutes_played >= min_minutes
    ]
    if verbose:
        _log(f"  · {len(eligible)} jogadores com ≥{min_minutes} min — extraindo passes …")

    player_passes: dict[int, tuple] = {}
    players_fetched = 0
    for idx, entry in enumerate(eligible, start=1):
        players_fetched += 1
        if verbose:
            _log(f"    [{idx}/{len(eligible)}] {entry.player.name}")
        try:
            actions = client.player_actions(event_id, entry.player.id)
        except NotFoundError:
            if verbose:
                _log("      (sem rating-breakdown)")
            continue
        if not actions.passes:
            if verbose:
                _log("      (0 passes)")
            continue
        player_passes[entry.player.id] = (entry, actions)
        if verbose:
            _log(f"      → {len(actions.passes)} ações de passe")

    mean_y_by_player: dict[int, float] = {}
    for pid, (_, actions) in player_passes.items():
        ys = [float(a.start_y) for a in actions.passes if a.start_y is not None]
        if ys:
            mean_y_by_player[pid] = _median(ys)

    position_by_id = resolve_match_positions(
        raw_by_player=raw_position_by_id,
        mean_y_by_player=mean_y_by_player,
    )

    rows: list[dict] = []
    players_with_passes = 0
    for pid, (entry, actions) in player_passes.items():
        players_with_passes += 1
        player_name = entry.player.name
        position = position_by_id.get(pid, raw_position_by_id.get(pid, ""))
        for action in actions.passes:
            if action.action_type not in PASS_ACTION_TYPES:
                continue
            rows.append(
                {
                    "category": "passes",
                    "eventActionType": action.action_type,
                    "isHome": action.is_home,
                    "outcome": action.outcome,
                    "keypass": action.is_keypass,
                    "isLongBall": "",
                    "start_x": action.start_x,
                    "start_y": action.start_y,
                    "end_x": action.end_x,
                    "end_y": action.end_y,
                    "player_id": pid,
                    "player_name": player_name,
                    "position": position,
                    **meta,
                }
            )
    return rows, players_with_passes, players_fetched


def _append_rows_csv(path: Path, rows: list[dict], *, columns: list[str]) -> None:
    if not rows:
        return
    write_header = not path.exists() or path.stat().st_size == 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _consolidate_match_files(out_dir: Path, out_name: str = "season_all.csv") -> int:
    import pandas as pd

    frames = [pd.read_csv(p) for p in sorted(out_dir.glob("match_*_passes.csv"))]
    if not frames:
        return 0
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(out_dir / out_name, index=False)
    return len(all_df)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch SofaScore pass coordinates only (fewer API calls)."
    )
    parser.add_argument("--url", help="SofaScore tournament URL with #id:SEASON")
    parser.add_argument("--tournament-id", type=int)
    parser.add_argument("--season-id", type=int)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=2.0,
        help="Seconds between API calls (default: 2.0 — safer for long seasons)",
    )
    parser.add_argument(
        "--proxy",
        default=None,
        metavar="URL",
        help="HTTPS proxy (or set TACOSCORE_PROXY / HTTPS_PROXY)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--consolidated-only",
        action="store_true",
        help="Append each match directly to season_all.csv (no per-match files)",
    )
    parser.add_argument(
        "--consolidate",
        action="store_true",
        help="After download, merge match_*_passes.csv into season_all.csv",
    )
    parser.add_argument(
        "--copy-season-to-root",
        action="store_true",
        help="Copy consolidated season_all.csv to the repo root for app.py",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--event-id", type=int, default=None)
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument(
        "--min-minutes",
        type=int,
        default=1,
        help="Skip players with fewer minutes (default: 1)",
    )
    parser.add_argument(
        "--cooldown-on-error",
        type=float,
        default=120.0,
        help="Seconds to sleep after HTTP 403 / rate-limit errors (default: 120)",
    )
    args = parser.parse_args()

    if args.url:
        tournament_id, season_id = parse_tournament_url(args.url)
    elif args.tournament_id is not None and args.season_id is not None:
        tournament_id, season_id = args.tournament_id, args.season_id
    else:
        parser.error("Provide --url or both --tournament-id and --season-id")

    out_dir = args.output_dir or (DEFAULT_OUT / f"{tournament_id}_{season_id}_passes")
    out_dir.mkdir(parents=True, exist_ok=True)
    done_path = out_dir / "done_passes.json"
    meta_path = out_dir / "metadata_passes.json"
    season_path = out_dir / "season_all.csv"

    try:
        from tacoscore import TacosScoreClient
        from tacoscore.exceptions import APIError
    except ImportError:
        print("Install: pip install -r requirements-sofascore.txt", file=sys.stderr)
        return 1

    proxies = _resolve_proxies(args.proxy)
    if proxies:
        _log(f"Proxy enabled · {_proxy_log_label(proxies)}")
    client = TacosScoreClient(rate_limit_seconds=args.rate_limit, proxies=proxies)

    _log(f"Listando jogos · tournament={tournament_id} season={season_id} …")
    matches = list_finished_matches_verbose(client, tournament_id, season_id)
    if args.event_id is not None:
        matches = [m for m in matches if m.event_id == args.event_id]
        if not matches:
            print(f"Event {args.event_id} not in finished matches.", file=sys.stderr)
            return 1
    elif args.limit:
        matches = matches[: args.limit]

    meta = {
        "mode": "passes_only",
        "tournament_id": tournament_id,
        "season_id": season_id,
        "listed_at": datetime.now(timezone.utc).isoformat(),
        "n_matches": len(matches),
        "requests_per_match_hint": "1 lineups + 1 rating-breakdown per player with minutes",
        "matches": [
            {
                "event_id": m.event_id,
                "date": m.start_timestamp.isoformat(),
                "home": m.home_team.name,
                "away": m.away_team.name,
                "score": m.display_score,
            }
            for m in matches
        ],
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"\n✓ {len(matches)} jogos finalizados para download → {meta_path}")

    if args.list_only:
        for m in matches:
            _log(f"  {m.event_id}  {m.start_timestamp:%Y-%m-%d}  {m}")
        return 0

    if args.consolidated_only and args.resume and season_path.exists():
        _log(f"Acrescentando em {season_path.name} existente …")

    done = load_done(done_path) if args.resume else set()
    ok = skipped = 0
    failed: list[tuple[int, str]] = []
    total_pass_rows = 0

    for i, summary in enumerate(matches, start=1):
        event_id = summary.event_id
        label = f"{summary.home_team.name} {summary.display_score} {summary.away_team.name}"
        if args.resume and event_id in done:
            skipped += 1
            _log(f"[{i}/{len(matches)}] SKIP · {event_id} · {label} (já em done_passes.json)")
            continue

        _log(f"\n[{i}/{len(matches)}] INÍCIO · {event_id} · {label}")
        try:
            rows, with_passes, fetched = fetch_match_passes(
                client,
                event_id,
                summary,
                min_minutes=args.min_minutes,
                verbose=True,
            )
            if args.consolidated_only:
                _append_rows_csv(season_path, rows, columns=ACTION_COLUMNS)
            else:
                import pandas as pd

                pd.DataFrame(rows, columns=ACTION_COLUMNS).to_csv(
                    out_dir / f"match_{event_id}_passes.csv",
                    index=False,
                )

            done.add(event_id)
            save_done(done_path, done)
            total_pass_rows += len(rows)
            _log(
                f"  ✓ FIM · {len(rows)} passes gravados · "
                f"{with_passes}/{fetched} jogadores com dados"
            )
            if len(rows) == 0:
                _log(
                    "  WARN: rating-breakdown sem passes — SofaScore pode não expor "
                    "coordenadas nesta competição/jogo."
                )
            ok += 1
        except APIError as exc:
            msg = f"{type(exc).__name__}: {exc}"
            _log(f"  ERROR: {msg}")
            failed.append((event_id, msg))
            if exc.status_code in (403, 429):
                _log(f"  Cooldown {args.cooldown_on_error:.0f}s antes de continuar …")
                time.sleep(args.cooldown_on_error)
            else:
                time.sleep(2.0)
        except Exception as exc:  # noqa: BLE001
            msg = f"{type(exc).__name__}: {exc}"
            _log(f"  ERROR: {msg}")
            failed.append((event_id, msg))
            time.sleep(2.0)

    _log(f"\nConcluído: {ok} baixados, {skipped} pulados, {len(failed)} falhas")
    if args.consolidated_only and season_path.exists():
        _log(f"Arquivo da temporada: {season_path} ({total_pass_rows:,} passes nesta execução)")

    if args.consolidate and not args.consolidated_only:
        n = _consolidate_match_files(out_dir)
        if n:
            _log(f"Consolidado {n} passes → {season_path}")

    if args.copy_season_to_root and season_path.exists():
        import pandas as pd

        dest = ROOT / "season_all.csv"
        pd.read_csv(season_path).to_csv(dest, index=False)
        _log(f"Copiado → {dest}")

    if failed:
        (out_dir / "failed_passes.json").write_text(
            json.dumps([{"event_id": eid, "error": err} for eid, err in failed], indent=2),
            encoding="utf-8",
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
