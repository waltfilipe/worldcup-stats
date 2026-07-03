#!/usr/bin/env python3
"""Download SofaScore data for a full tournament season.

Exports three datasets per match (and consolidated season files):

1. **Actions** (`season_all.csv`) — pass / carry / dribble / defensive coordinates
   with player position from the lineup.
2. **Player match stats** (`player_match_stats.csv`) — full SofaScore box-score
   per player (xG, xA, shots, tackles, passes, rating, …).
3. **Shots** (`season_shots.csv`) — shot-level xG / xGOT with coordinates.

Example::

    pip install -r requirements-sofascore.txt
    python scripts/fetch_sofascore_season.py \\
        --url "https://www.sofascore.com/football/tournament/world/world-championship/16#id:58210" \\
        --consolidate --resume

Run locally — Sofascore may return HTTP 403 from cloud IPs.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "sofascore"

# Wyscout-style action export (compatible with app.py)
ACTION_COLUMNS = [
    "category",
    "eventActionType",
    "isHome",
    "outcome",
    "keypass",
    "isLongBall",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "player_id",
    "player_name",
    "position",
    "event_id",
    "home_team",
    "away_team",
    "match_date",
]

# SofaScore G/D/M/F → short codes used in heuristic_scoring.py
POSITION_TO_APP = {
    "G": "GK",
    "D": "CB",
    "M": "CM",
    "F": "ST",
}

ACTION_CATEGORY_MAP: dict[str, tuple[str, str]] = {
    "pass": ("passes", "pass"),
    "cross": ("passes", "cross"),
    "throw-in": ("passes", "throw-in"),
    "ball-carry": ("ball-carries", "ball-carry"),
    "dribble": ("dribbles", "dribble"),
    "tackle": ("defensive", "tackle"),
    "ball-recovery": ("defensive", "ball-recovery"),
}


def parse_tournament_url(url: str) -> tuple[int, int]:
    url = url.strip()
    if "#id:" not in url:
        raise ValueError(
            "URL must include the season fragment, e.g. "
            "'.../world-championship/16#id:58210'"
        )
    path_part, frag = url.split("#id:", 1)
    season_id = int(frag.split("&")[0].split("/")[0].strip())
    path_clean = path_part.rstrip("/").split("?")[0]
    tournament_match = re.search(r"/(\d+)$", path_clean)
    if not tournament_match:
        raise ValueError(f"Could not parse tournament id from URL path: {path_part}")
    return int(tournament_match.group(1)), season_id


def list_finished_matches(client, tournament_id: int, season_id: int):
    by_round = client.season_events(tournament_id, season_id)
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


def _lineup_context(match) -> tuple[dict[int, str], dict[int, str], dict[int, Any]]:
    """player_id → name, position (G/D/M/F), lineup entry."""
    name_by_id: dict[int, str] = {}
    position_by_id: dict[int, str] = {}
    entry_by_id: dict[int, Any] = {}
    for entry in match.lineups.all_players():
        pid = entry.player.id
        name_by_id[pid] = entry.player.name
        position_by_id[pid] = entry.position_match or entry.player.position or ""
        entry_by_id[pid] = entry
    return name_by_id, position_by_id, entry_by_id


def _match_meta(match) -> dict[str, Any]:
    summary = match.event_detail.summary
    return {
        "event_id": match.event_id,
        "home_team": summary.home_team.name,
        "away_team": summary.away_team.name,
        "match_date": summary.start_timestamp.isoformat(),
    }


def _action_rows(
    match,
    *,
    categories: set[str],
    name_by_id: dict[int, str],
    position_by_id: dict[int, str],
) -> list[dict]:
    meta = _match_meta(match)
    rows: list[dict] = []
    stream_map = {
        "passes": lambda a: a.passes,
        "ball-carries": lambda a: a.ball_carries,
        "dribbles": lambda a: a.dribbles,
        "defensive": lambda a: a.defensive,
    }

    for pid, pdata in match.player_data.items():
        if not pdata.actions:
            continue
        player_name = name_by_id.get(pid, str(pid))
        position = position_by_id.get(pid, "")
        for stream_key, getter in stream_map.items():
            if stream_key not in categories:
                continue
            for action in getter(pdata.actions):
                cat_label, event_type = ACTION_CATEGORY_MAP.get(
                    action.action_type,
                    (stream_key, action.action_type),
                )
                rows.append(
                    {
                        "category": cat_label,
                        "eventActionType": event_type,
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
    return rows


def _player_stats_rows(match, *, position_by_id: dict[int, str]) -> list[dict]:
    """One row per player in the lineup with the full SofaScore statistics block."""
    meta = _match_meta(match)
    rows: list[dict] = []
    for side_name, side in [("home", match.lineups.home), ("away", match.lineups.away)]:
        for entry in side.players:
            stats_dict = dataclasses.asdict(entry.statistics)
            raw_position = entry.position_match or entry.player.position or ""
            rows.append(
                {
                    **meta,
                    "player_id": entry.player.id,
                    "player_name": entry.player.name,
                    "position": raw_position,
                    "position_app": POSITION_TO_APP.get(raw_position, raw_position),
                    "position_career": entry.player.position,
                    "shirt_number": entry.shirt_number,
                    "is_substitute": entry.is_substitute,
                    "is_captain": entry.is_captain,
                    "is_home": side_name == "home",
                    "side": side_name,
                    "formation": side.formation,
                    **stats_dict,
                }
            )
    return rows


def _shot_rows(
    match,
    *,
    name_by_id: dict[int, str],
    position_by_id: dict[int, str],
) -> list[dict]:
    """Shot-level xG from the match shotmap (all players)."""
    meta = _match_meta(match)
    shotmap = match.match_shotmap
    if not shotmap or not shotmap.shots:
        return []

    rows: list[dict] = []
    for shot in shotmap.shots:
        pid = shot.player_id
        pc = shot.player_coords
        gmc = shot.goal_mouth_coords
        bc = shot.block_coords
        rows.append(
            {
                **meta,
                "player_id": pid,
                "player_name": name_by_id.get(pid, str(pid)),
                "position": position_by_id.get(pid, ""),
                "shot_id": shot.id,
                "goalkeeper_id": shot.goalkeeper_id,
                "is_home": shot.is_home,
                "shot_type": shot.shot_type,
                "goal_type": shot.goal_type,
                "situation": shot.situation,
                "body_part": shot.body_part,
                "goal_mouth_location": shot.goal_mouth_location,
                "xg": shot.xg,
                "xgot": shot.xgot,
                "minute": shot.minute,
                "added_time": shot.added_time,
                "time_seconds": shot.time_seconds,
                "player_x": pc.x,
                "player_y": pc.y,
                "player_z": pc.z,
                "goal_mouth_x": gmc.x if gmc else None,
                "goal_mouth_y": gmc.y if gmc else None,
                "goal_mouth_z": gmc.z if gmc else None,
                "block_x": bc.x if bc else None,
                "block_y": bc.y if bc else None,
                "block_z": bc.z if bc else None,
            }
        )
    return rows


def fetch_match_export(
    client,
    event_id: int,
    *,
    categories: set[str],
) -> dict[str, list[dict]]:
    """Fetch one match and return actions, player stats, and shots."""
    match = client.fetch_full_match(event_id)
    if not match.event_detail:
        raise RuntimeError(f"event {event_id}: missing event_detail")

    name_by_id, position_by_id, _ = _lineup_context(match)
    return {
        "actions": _action_rows(
            match,
            categories=categories,
            name_by_id=name_by_id,
            position_by_id=position_by_id,
        ),
        "player_stats": _player_stats_rows(match, position_by_id=position_by_id),
        "shots": _shot_rows(match, name_by_id=name_by_id, position_by_id=position_by_id),
    }


def load_done(path: Path) -> set[int]:
    if not path.exists():
        return set()
    return set(json.loads(path.read_text(encoding="utf-8")))


def save_done(path: Path, done: set[int]) -> None:
    path.write_text(json.dumps(sorted(done), indent=2), encoding="utf-8")


def _write_csv(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _consolidate_glob(out_dir: Path, pattern: str, out_name: str) -> int:
    import pandas as pd

    frames = [pd.read_csv(p) for p in sorted(out_dir.glob(pattern))]
    if not frames:
        return 0
    all_df = pd.concat(frames, ignore_index=True)
    _write_csv(all_df, out_dir / out_name)
    return len(all_df)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch SofaScore coordinates, player stats, and shots for a tournament."
    )
    parser.add_argument("--url", help="SofaScore tournament URL with #id:SEASON")
    parser.add_argument("--tournament-id", type=int)
    parser.add_argument("--season-id", type=int)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["passes", "ball-carries"],
        choices=["passes", "ball-carries", "dribbles", "defensive"],
        help="Action streams to export (default: passes ball-carries)",
    )
    parser.add_argument("--rate-limit", type=float, default=0.4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--consolidate", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument(
        "--copy-season-to-root",
        action="store_true",
        help="Also write consolidated season_all.csv to the repo root (for app.py)",
    )
    args = parser.parse_args()

    if args.url:
        tournament_id, season_id = parse_tournament_url(args.url)
    elif args.tournament_id is not None and args.season_id is not None:
        tournament_id, season_id = args.tournament_id, args.season_id
    else:
        parser.error("Provide --url or both --tournament-id and --season-id")

    out_dir = args.output_dir or (DEFAULT_OUT / f"{tournament_id}_{season_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    done_path = out_dir / "done.json"
    meta_path = out_dir / "metadata.json"
    categories = set(args.categories)

    try:
        from tacoscore import TacosScoreClient
    except ImportError:
        print("Install: pip install -r requirements-sofascore.txt", file=sys.stderr)
        return 1

    import pandas as pd

    client = TacosScoreClient(rate_limit_seconds=args.rate_limit)
    print(f"Listing matches · tournament={tournament_id} season={season_id} …")
    matches = list_finished_matches(client, tournament_id, season_id)
    if args.limit:
        matches = matches[: args.limit]

    meta = {
        "tournament_id": tournament_id,
        "season_id": season_id,
        "listed_at": datetime.now(timezone.utc).isoformat(),
        "n_matches": len(matches),
        "categories": sorted(categories),
        "exports": ["actions", "player_stats", "shots"],
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
    print(f"Found {len(matches)} finished matches → {meta_path}")

    if args.list_only:
        for m in matches:
            print(f"  {m.event_id}  {m.start_timestamp:%Y-%m-%d}  {m}")
        return 0

    done = load_done(done_path) if args.resume else set()
    ok = skipped = 0
    failed: list[tuple[int, str]] = []

    for i, summary in enumerate(matches, start=1):
        event_id = summary.event_id
        label = f"{summary.home_team.name} {summary.display_score} {summary.away_team.name}"
        if args.resume and event_id in done:
            skipped += 1
            continue

        print(f"[{i}/{len(matches)}] {event_id} · {label}")
        try:
            payload = fetch_match_export(client, event_id, categories=categories)

            actions_df = pd.DataFrame(payload["actions"], columns=ACTION_COLUMNS)
            stats_df = pd.DataFrame(payload["player_stats"])
            shots_df = pd.DataFrame(payload["shots"])

            _write_csv(actions_df, out_dir / f"match_{event_id}_actions.csv")
            _write_csv(stats_df, out_dir / f"match_{event_id}_player_stats.csv")
            _write_csv(shots_df, out_dir / f"match_{event_id}_shots.csv")

            done.add(event_id)
            save_done(done_path, done)
            print(
                f"  → {len(actions_df)} actions, {len(stats_df)} player-rows, "
                f"{len(shots_df)} shots"
            )
            ok += 1
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            print(f"  ERROR: {msg}")
            failed.append((event_id, msg))
            time.sleep(2.0)

    print(f"\nDone: {ok} downloaded, {skipped} skipped, {len(failed)} failed")

    if args.consolidate:
        n_actions = _consolidate_glob(out_dir, "match_*_actions.csv", "season_all.csv")
        n_stats = _consolidate_glob(out_dir, "match_*_player_stats.csv", "player_match_stats.csv")
        n_shots = _consolidate_glob(out_dir, "match_*_shots.csv", "season_shots.csv")
        if n_actions:
            print(f"Consolidated {n_actions} action rows → {out_dir / 'season_all.csv'}")
        if n_stats:
            print(f"Consolidated {n_stats} player-rows → {out_dir / 'player_match_stats.csv'}")
        if n_shots:
            print(f"Consolidated {n_shots} shots → {out_dir / 'season_shots.csv'}")
        if args.copy_season_to_root and n_actions:
            season_path = ROOT / "season_all.csv"
            pd.read_csv(out_dir / "season_all.csv").to_csv(season_path, index=False)
            print(f"Copied season_all.csv → {season_path}")

    if failed:
        (out_dir / "failed.json").write_text(
            json.dumps([{"event_id": eid, "error": err} for eid, err in failed], indent=2),
            encoding="utf-8",
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
