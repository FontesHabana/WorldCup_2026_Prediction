from typing import List
from src.simulation.models import GroupResult, TeamStats, MatchResult
from src.simulation.match_simulator import simulate_match
import random


def get_best_third_places(group_results: List[GroupResult]) -> List[TeamStats]:
    """
    Takes the results from all 12 groups and returns the 8 best 3rd-place teams.
    """
    all_thirds = []
    for grp in group_results:
        # Assuming standings are already sorted, the 3rd team is at index 2
        all_thirds.append(grp.standings[2])

    # Sort all 3rd place teams by Points, then GD, then GF
    # We can use the same logic we used for group standings!
    # For now, a simple sort:
    all_thirds.sort(key=lambda x: (x.points, x.goal_difference, x.goals_for), reverse=True)

    return all_thirds[:8]


def simulate_knockout_match(home: str, away: str, model_params: dict):
    """
    Runs a match. If it's a draw, it forces a winner.
    Returns the NAME of the winning team.
    """
    res = simulate_match(home, away, model_params, neutral=True)



    if res.winner != "Draw":
        return res.winner, res
    else:
        # Penalty Shootout (Current: 50/50 baseline)

        return random.choice([home, away]),res