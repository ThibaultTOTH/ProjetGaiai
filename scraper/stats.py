"""
Statistics and dataset analysis for the Gaia Project scraper.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Any, List

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.box import ASCII
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from .config import BASE_FACTIONS, METADATA_FILE


def load_dataset_metadata(meta_path: Path = METADATA_FILE) -> Dict[str, Any]:
    if not meta_path.exists():
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_stats():
    data = load_dataset_metadata()
    if not data or not data.get("games"):
        print(f"No games found in metadata file at {METADATA_FILE}.")
        print("Run 'python -m scraper.scraper' to start collecting games.")
        return

    games = data.get("games", {})
    total_games = len(games)

    faction_stats = defaultdict(lambda: {
        "count": 0,
        "wins": 0,
        "scores": [],
        "elos": [],
    })

    player_stats = defaultdict(lambda: {
        "games": 0,
        "wins": 0,
        "scores": [],
        "elos": [],
    })

    total_moves = 0
    winning_scores = []

    for gid, g in games.items():
        total_moves += g.get("num_moves", 0)
        players = g.get("players", [])
        if not players:
            continue

        # Determine winner
        best_score = max(p.get("score", 0) for p in players)
        winning_scores.append(best_score)

        for p in players:
            f = p.get("faction")
            score = p.get("score", 0)
            name = p.get("name") or "Unknown"
            elo = p.get("elo")

            if f:
                faction_stats[f]["count"] += 1
                faction_stats[f]["scores"].append(score)
                if elo:
                    faction_stats[f]["elos"].append(elo)
                if score == best_score:
                    faction_stats[f]["wins"] += 1

            player_stats[name]["games"] += 1
            player_stats[name]["scores"].append(score)
            if elo:
                player_stats[name]["elos"].append(elo)
            if score == best_score:
                player_stats[name]["wins"] += 1

    avg_win_vp = sum(winning_scores) / len(winning_scores) if winning_scores else 0
    max_vp_overall = max(winning_scores) if winning_scores else 0

    if HAS_RICH:
        console = Console(force_terminal=True, legacy_windows=False)
        console.print(Panel.fit(
            f"[bold green]Gaia Project Expert Dataset Summary[/bold green]\n"
            f"Total Games: [cyan]{total_games:,}[/cyan] | "
            f"Total Moves: [cyan]{total_moves:,}[/cyan] | "
            f"Avg Winning Score: [yellow]{avg_win_vp:.1f} VP[/yellow] | "
            f"Max Score: [magenta]{max_vp_overall} VP[/magenta]",
            border_style="blue",
            box=ASCII
        ))

        # Faction breakdown table
        table = Table(title="Faction Balance & Performance", box=ASCII)
        table.add_column("Faction", justify="left", style="cyan")
        table.add_column("Games", justify="right", style="green")
        table.add_column("Share %", justify="right", style="white")
        table.add_column("Win Rate", justify="right", style="yellow")
        table.add_column("Avg VP", justify="right", style="magenta")
        table.add_column("Max VP", justify="right", style="red")

        total_faction_picks = sum(s["count"] for s in faction_stats.values())
        for f in sorted(BASE_FACTIONS):
            s = faction_stats[f]
            count = s["count"]
            share = (count / total_faction_picks * 100) if total_faction_picks else 0.0
            win_rate = (s["wins"] / count * 100) if count else 0.0
            avg_vp = (sum(s["scores"]) / count) if count else 0.0
            max_f_vp = max(s["scores"]) if s["scores"] else 0

            table.add_row(
                f,
                f"{count:,}",
                f"{share:4.1f}%",
                f"{win_rate:4.1f}%",
                f"{avg_vp:5.1f}",
                str(max_f_vp),
            )

        console.print(table)

        # Top 10 players
        top_players = sorted(player_stats.items(), key=lambda x: x[1]["games"], reverse=True)[:10]
        p_table = Table(title="Top 10 Most Frequent Players in Dataset", box=ASCII)
        p_table.add_column("Player", justify="left", style="cyan")
        p_table.add_column("Games", justify="right", style="green")
        p_table.add_column("Wins", justify="right", style="yellow")
        p_table.add_column("Win Rate", justify="right", style="white")
        p_table.add_column("Avg Score", justify="right", style="magenta")
        p_table.add_column("Avg Elo", justify="right", style="blue")

        for name, ps in top_players:
            count = ps["games"]
            wins = ps["wins"]
            wr = (wins / count * 100) if count else 0.0
            avg_s = (sum(ps["scores"]) / count) if count else 0.0
            avg_e = (sum(ps["elos"]) / len(ps["elos"])) if ps["elos"] else 0.0
            p_table.add_row(name, str(count), str(wins), f"{wr:4.1f}%", f"{avg_s:.1f}", f"{avg_e:.0f}" if avg_e else "N/A")

        console.print(p_table)

    else:
        print("\n=== Gaia Project Expert Dataset Summary ===")
        print(f"Total Games: {total_games} | Total Moves: {total_moves} | Avg Winning Score: {avg_win_vp:.1f} VP | Max Score: {max_vp_overall} VP\n")
        print(f"{'Faction':16s} | {'Games':6s} | {'WinRate':7s} | {'Avg VP':6s} | {'Max VP':6s}")
        print("-" * 52)
        for f in sorted(BASE_FACTIONS):
            s = faction_stats[f]
            count = s["count"]
            win_rate = (s["wins"] / count * 100) if count else 0.0
            avg_vp = (sum(s["scores"]) / count) if count else 0.0
            max_f_vp = max(s["scores"]) if s["scores"] else 0
            print(f"{f:16s} | {count:6d} | {win_rate:6.1f}% | {avg_vp:6.1f} | {max_f_vp:6d}")


if __name__ == "__main__":
    print_stats()
