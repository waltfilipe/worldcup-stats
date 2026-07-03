#!/usr/bin/env python3
"""Test whether SofaScore exposes rating-breakdown (pass/carry coordinates) for a match."""

from __future__ import annotations

import argparse
import os
import sys


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose action coverage for one SofaScore event.")
    parser.add_argument("event_id", type=int)
    parser.add_argument(
        "--proxy",
        default=None,
        metavar="URL",
        help="HTTPS proxy URL (or set TACOSCORE_PROXY / HTTPS_PROXY)",
    )
    args = parser.parse_args()

    try:
        from tacoscore import TacosScoreClient
    except ImportError:
        print("pip install -r requirements-sofascore.txt", file=sys.stderr)
        return 1

    proxies = _resolve_proxies(args.proxy)
    client = TacosScoreClient(rate_limit_seconds=0.4, proxies=proxies)
    eid = args.event_id
    match = client.fetch_full_match(eid)

    print(f"Event {eid}")
    if match.event_detail:
        s = match.event_detail.summary
        print(f"  {s.home_team.name} {s.display_score} {s.away_team.name}")

    with_actions = 0
    with_heatmap = 0
    sample_pid = None
    sample_actions = None

    for pid, pdata in match.player_data.items():
        n = 0
        if pdata.actions:
            n = (
                len(pdata.actions.passes)
                + len(pdata.actions.ball_carries)
                + len(pdata.actions.dribbles)
                + len(pdata.actions.defensive)
            )
        if n > 0:
            with_actions += 1
            if sample_pid is None:
                sample_pid = pid
                sample_actions = pdata.actions
        if pdata.heatmap and pdata.heatmap.points:
            with_heatmap += 1

    total = len(match.player_data)
    shots = len(match.match_shotmap.shots) if match.match_shotmap else 0

    print(f"\nPlayers in spatial fetch: {total}")
    print(f"Players with rating-breakdown actions: {with_actions}")
    print(f"Players with heatmap touches: {with_heatmap}")
    print(f"Shots (match shotmap): {shots}")

    if with_actions == 0:
        print(
            "\nConclusão: rating-breakdown VAZIO para este jogo/competição.\n"
            "Passes/carries com coordenadas NÃO estão disponíveis no SofaScore.\n"
            "Você ainda tem: player_match_stats (contagens) e season_shots (chutes)."
        )
    else:
        a = sample_actions
        print(
            f"\nExemplo jogador {sample_pid}: "
            f"{len(a.passes)} passes, {len(a.ball_carries)} carries, "
            f"{len(a.defensive)} defensive"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
