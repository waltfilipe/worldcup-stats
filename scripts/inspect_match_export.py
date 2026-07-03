#!/usr/bin/env python3
"""Quick sanity check for one match export directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize one SofaScore match export.")
    parser.add_argument("event_id", type=int, help="SofaScore event id")
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="Directory with match_{id}_*.csv (default: data/sofascore/*/ )",
    )
    args = parser.parse_args()

    eid = args.event_id
    if args.dir:
        base = args.dir
    else:
        root = Path(__file__).resolve().parent.parent / "data" / "sofascore"
        candidates = list(root.glob(f"*/match_{eid}_actions.csv"))
        if not candidates:
            print(f"No export found for event {eid} under {root}")
            return 1
        base = candidates[0].parent

    actions = base / f"match_{eid}_actions.csv"
    stats = base / f"match_{eid}_player_stats.csv"
    shots = base / f"match_{eid}_shots.csv"

    print(f"Directory: {base}\n")

    if actions.exists():
        df = pd.read_csv(actions)
        print(f"=== Actions ({len(df)} rows) ===")
        print(df.groupby(["category", "eventActionType"]).size().to_string())
        if "position" in df.columns:
            print(f"\nPlayers with position: {df['player_name'].nunique()}")
            print(df.groupby("position")["player_name"].nunique().head(10).to_string())
        defensive = df[df["category"] == "defensive"]
        if not defensive.empty:
            print(f"\nDefensive sample (up to 5):")
            cols = ["player_name", "eventActionType", "start_x", "start_y", "outcome"]
            print(defensive[cols].head().to_string(index=False))
    else:
        print(f"Missing: {actions.name}")

    if stats.exists():
        st = pd.read_csv(stats)
        print(f"\n=== Player stats ({len(st)} rows) ===")
        def_cols = [
            "total_tackle", "won_tackle", "interception_won", "total_clearance",
            "outfielder_block", "ball_recovery", "expected_goals", "total_shots",
        ]
        present = [c for c in def_cols if c in st.columns]
        totals = st[present].sum(numeric_only=True)
        print(totals.to_string())
        top = st.nlargest(5, "total_tackle" if "total_tackle" in st.columns else present[0])
        show = ["player_name", "position"] + present
        print(f"\nTop by tackles:\n{top[show].to_string(index=False)}")
    else:
        print(f"\nMissing: {stats.name}")

    if shots.exists():
        sh = pd.read_csv(shots)
        print(f"\n=== Shots ({len(sh)} rows) ===")
        if "shot_type" in sh.columns:
            print(sh["shot_type"].value_counts().to_string())
        if "xg" in sh.columns:
            print(f"Total xG: {sh['xg'].sum():.3f}")
    else:
        print(f"\nMissing: {shots.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
