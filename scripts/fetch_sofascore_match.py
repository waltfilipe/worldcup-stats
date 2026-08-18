#!/usr/bin/env python3
"""Export all available SofaScore data for a single match.

Outputs (under --output-dir, default data/sofascore/match_{event_id}/):

  match_{id}_actions.csv       passes, carries, dribbles, defensive (+ heatmap fallback)
  match_{id}_player_stats.csv  full box score per player
  match_{id}_shots.csv         shotmap with xG / coordinates
  match_{id}_heatmap.csv       per-player touch heatmap points
  match_{id}_team_stats.csv    team-level statistics (long format)
  match_{id}_incidents.json    goals, cards, substitutions, periods
  match_{id}_graph.json        match momentum graph
  match_{id}_summary.json      coverage report

Example::

    pip install -r requirements-sofascore.txt
    python -u scripts/fetch_sofascore_match.py 12813008 --output-dir ./allmatch
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for _path in (SCRIPT_DIR, ROOT):
    _s = str(_path)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from fetch_sofascore_season import (  # noqa: E402
    ACTION_COLUMNS,
    _action_rows,
    _heatmap_rows,
    _lineup_context,
    _match_meta,
    _player_stats_rows,
    _shot_rows,
)

VERBOSE = True


def _log(message: str = "") -> None:
    if VERBOSE:
        print(message, flush=True)


def parse_event_id(value: str) -> int:
    value = value.strip()
    if value.isdigit():
        return int(value)
    match = re.search(r"/(?:event|match|football/match)/[^/]*/(\d+)", value)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d{6,})", value)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not parse SofaScore event id from: {value!r}")


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


def _count_player_actions(actions) -> tuple[int, int, int, int]:
    if not actions:
        return 0, 0, 0, 0
    return (
        len(actions.passes),
        len(actions.ball_carries),
        len(actions.dribbles),
        len(actions.defensive),
    )


def fetch_match_verbose(client, event_id: int) -> Any:
    """Same data as fetch_full_match, logging each API step and player."""
    from tacoscore.exceptions import NotFoundError
    from tacoscore.extraction import (
        should_fetch_heatmap,
        should_fetch_shotmap,
        should_fetch_spatial,
    )
    from tacoscore.models.event import FullMatch, PlayerMatchData

    _log(f"[1/4] Metadados do jogo (event {event_id}) …")
    event_detail = client.event(event_id)
    if event_detail and event_detail.summary:
        s = event_detail.summary
        _log(f"      {s.home_team.name} {s.display_score} {s.away_team.name}")

    _log("      · lineups")
    lineups = client.event_lineups(event_id)
    _log("      · team statistics")
    team_stats = client.event_statistics(event_id)

    _log("      · incidents / managers / graph / h2h / pregame / shotmap")
    incidents = managers = graph = head_to_head = pregame_form = match_shotmap = None
    with contextlib.suppress(NotFoundError):
        incidents = client.event_incidents(event_id)
        _log(f"        incidents: {len(incidents.items)} entradas")
    with contextlib.suppress(NotFoundError):
        managers = client.event_managers(event_id)
    with contextlib.suppress(NotFoundError):
        graph = client.event_graph(event_id)
    with contextlib.suppress(NotFoundError):
        head_to_head = client.event_h2h(event_id)
    with contextlib.suppress(NotFoundError):
        pregame_form = client.event_pregame_form(event_id)
    with contextlib.suppress(NotFoundError):
        match_shotmap = client.event_shotmap(event_id)
        n_shots = len(match_shotmap.shots) if match_shotmap else 0
        _log(f"        shotmap: {n_shots} chutes")

    players = list(lineups.all_players())
    eligible = [
        e
        for e in players
        if should_fetch_spatial(e.statistics.minutes_played, skip_no_minutes=True)
    ]
    _log(
        f"[2/4] Dados por jogador · {len(eligible)}/{len(players)} "
        f"com minutos (rating-breakdown + heatmap) …"
    )

    player_data: dict[int, PlayerMatchData] = {}
    for i, entry in enumerate(eligible, start=1):
        stats = entry.statistics
        name = entry.player.name
        _log(f"  [{i}/{len(eligible)}] {name} ({stats.minutes_played} min) …")

        heatmap = actions = shotmap = None
        if should_fetch_heatmap(stats.touches, skip_sparse=True, min_touches=5):
            with contextlib.suppress(NotFoundError):
                heatmap = client.player_heatmap(event_id, entry.player.id)
                if heatmap and heatmap.points:
                    _log(f"        heatmap: {len(heatmap.points)} toques")

        with contextlib.suppress(NotFoundError):
            actions = client.player_actions(event_id, entry.player.id)
            if actions:
                np, nc, nd, nf = _count_player_actions(actions)
                total = np + nc + nd + nf
                if total:
                    _log(
                        f"        ações: {total} "
                        f"(pass {np} · carry {nc} · dribble {nd} · def {nf})"
                    )
                else:
                    _log("        ações: 0 (rating-breakdown vazio)")

        if should_fetch_shotmap(stats.total_shots):
            with contextlib.suppress(NotFoundError):
                shotmap = client.player_shotmap(event_id, entry.player.id)
                if shotmap and shotmap.shots:
                    _log(f"        chutes jogador: {len(shotmap.shots)}")

        player_data[entry.player.id] = PlayerMatchData(
            player_id=entry.player.id,
            heatmap=heatmap,
            actions=actions,
            shotmap=shotmap,
        )

    _log("[3/4] Montando tabelas CSV …")
    return FullMatch(
        event_id=event_id,
        team_statistics=team_stats,
        lineups=lineups,
        player_data=player_data,
        event_detail=event_detail,
        incidents=incidents,
        graph=graph,
        managers=managers,
        head_to_head=head_to_head,
        pregame_form=pregame_form,
        match_shotmap=match_shotmap,
    )


def _team_stats_rows(match) -> list[dict]:
    meta = _match_meta(match)
    ts = match.team_statistics
    if not ts or not getattr(ts, "by_period", None):
        return []
    rows: list[dict] = []
    for period, keys in ts.by_period.items():
        for key, val in keys.items():
            rows.append(
                {
                    **meta,
                    "period": period,
                    "stat_key": key,
                    "home_value": val.home_value,
                    "away_value": val.away_value,
                    "home_total": val.home_total,
                    "away_total": val.away_total,
                }
            )
    return rows


def _heatmap_only_rows(match, *, name_by_id: dict, position_by_id: dict) -> list[dict]:
    meta = _match_meta(match)
    rows: list[dict] = []
    for pid, pdata in match.player_data.items():
        if not pdata.heatmap or not pdata.heatmap.points:
            continue
        for pt in pdata.heatmap.points:
            rows.append(
                {
                    **meta,
                    "player_id": pid,
                    "player_name": name_by_id.get(pid, str(pid)),
                    "position": position_by_id.get(pid, ""),
                    "x": pt.x,
                    "y": pt.y,
                    "count": getattr(pt, "count", 1),
                }
            )
    return rows


def _dataclass_to_json(obj: Any) -> Any:
    if obj is None:
        return None
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if isinstance(obj, list):
        return [_dataclass_to_json(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _dataclass_to_json(v) for k, v in obj.items()}
    return obj


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export all available SofaScore data for one match."
    )
    parser.add_argument("event", help="SofaScore event id or match URL")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--rate-limit", type=float, default=0.4)
    parser.add_argument("--proxy", default=None, metavar="URL")
    parser.add_argument("--no-heatmap-fallback", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Hide progress log")
    args = parser.parse_args()

    global VERBOSE
    VERBOSE = not args.quiet

    try:
        event_id = parse_event_id(args.event)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    try:
        from tacoscore import TacosScoreClient
    except ImportError:
        print("pip install -r requirements-sofascore.txt", file=sys.stderr)
        return 1

    import pandas as pd

    out_dir = args.output_dir or (ROOT / "data" / "sofascore" / f"match_{event_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    client = TacosScoreClient(
        rate_limit_seconds=args.rate_limit,
        proxies=_resolve_proxies(args.proxy),
    )

    _log(f"Iniciando · event_id={event_id} · saída={out_dir.resolve()}")
    match = fetch_match_verbose(client, event_id)

    if not match.event_detail:
        print(f"Event {event_id}: missing event_detail", file=sys.stderr)
        return 1

    name_by_id, position_by_id, _ = _lineup_context(match)
    raw_position_by_id = {
        entry.player.id: entry.position_match or entry.player.position or ""
        for entry in match.lineups.all_players()
    }
    categories = {"passes", "ball-carries", "dribbles", "defensive"}

    actions = _action_rows(
        match,
        categories=categories,
        name_by_id=name_by_id,
        position_by_id=position_by_id,
    )
    used_heatmap_fallback = False
    if not actions and not args.no_heatmap_fallback:
        actions = _heatmap_rows(
            match, name_by_id=name_by_id, position_by_id=position_by_id
        )
        used_heatmap_fallback = bool(actions)

    player_stats = _player_stats_rows(
        match,
        position_by_id=position_by_id,
        raw_position_by_id=raw_position_by_id,
    )
    shots = _shot_rows(match, name_by_id=name_by_id, position_by_id=position_by_id)
    heatmap = _heatmap_only_rows(match, name_by_id=name_by_id, position_by_id=position_by_id)
    team_stats = _team_stats_rows(match)

    prefix = f"match_{event_id}"
    actions_df = (
        pd.DataFrame(actions, columns=ACTION_COLUMNS)
        if actions
        else pd.DataFrame(columns=ACTION_COLUMNS)
    )
    stats_df = pd.DataFrame(player_stats)
    shots_df = pd.DataFrame(shots)
    heatmap_df = pd.DataFrame(heatmap)
    team_df = pd.DataFrame(team_stats)

    actions_df.to_csv(out_dir / f"{prefix}_actions.csv", index=False)
    _log(f"  ✓ {prefix}_actions.csv ({len(actions_df)} linhas)")
    stats_df.to_csv(out_dir / f"{prefix}_player_stats.csv", index=False)
    _log(f"  ✓ {prefix}_player_stats.csv ({len(stats_df)} linhas)")
    shots_df.to_csv(out_dir / f"{prefix}_shots.csv", index=False)
    _log(f"  ✓ {prefix}_shots.csv ({len(shots_df)} linhas)")
    heatmap_df.to_csv(out_dir / f"{prefix}_heatmap.csv", index=False)
    _log(f"  ✓ {prefix}_heatmap.csv ({len(heatmap_df)} linhas)")
    team_df.to_csv(out_dir / f"{prefix}_team_stats.csv", index=False)
    _log(f"  ✓ {prefix}_team_stats.csv ({len(team_df)} linhas)")

    (out_dir / f"{prefix}_incidents.json").write_text(
        json.dumps(_dataclass_to_json(match.incidents), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _log(f"  ✓ {prefix}_incidents.json")
    (out_dir / f"{prefix}_graph.json").write_text(
        json.dumps(_dataclass_to_json(match.graph), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _log(f"  ✓ {prefix}_graph.json")

    with_actions = sum(
        1
        for pdata in match.player_data.values()
        if pdata.actions
        and (
            len(pdata.actions.passes)
            + len(pdata.actions.ball_carries)
            + len(pdata.actions.dribbles)
            + len(pdata.actions.defensive)
        )
        > 0
    )
    with_heatmap = sum(
        1 for pdata in match.player_data.values() if pdata.heatmap and pdata.heatmap.points
    )

    meta = _match_meta(match)
    summary = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "event_id": event_id,
        **meta,
        "counts": {
            "actions": len(actions_df),
            "player_stats_rows": len(stats_df),
            "shots": len(shots_df),
            "heatmap_points": len(heatmap_df),
            "team_stats_rows": len(team_df),
        },
        "action_breakdown": (
            {
                f"{cat}|{etype}": int(n)
                for (cat, etype), n in actions_df.groupby(["category", "eventActionType"]).size().items()
            }
            if not actions_df.empty
            else {}
        ),
        "coverage": {
            "players_in_spatial_fetch": len(match.player_data),
            "players_with_rating_breakdown": with_actions,
            "players_with_heatmap": with_heatmap,
            "used_heatmap_fallback": used_heatmap_fallback,
        },
    }
    summary_path = out_dir / f"{prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    _log(f"\n[4/4] {meta['home_team']} {match.event_detail.summary.display_score} {meta['away_team']}")
    _log(f"Pasta: {out_dir.resolve()}")
    _log(
        f"Total: {len(actions_df)} ações · {len(stats_df)} jogadores · "
        f"{len(shots_df)} chutes · {len(heatmap_df)} heatmap"
    )
    if len(actions_df) == 0:
        _log(
            "AVISO: rating-breakdown vazio — stats/chutes/heatmap seguem disponíveis."
        )
    elif used_heatmap_fallback:
        _log("NOTA: ações vieram do heatmap (só toques).")
    _log("Concluído.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
