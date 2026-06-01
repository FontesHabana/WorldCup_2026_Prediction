from typing import List, Dict, Tuple
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



def run_tournament_knockout(group_results: List[GroupResult], model_params: dict):
    # 1. Identify all qualified teams
    winners = {gr.group_name: gr.standings[0].name for gr in group_results}
    runners = {gr.group_name: gr.standings[1].name for gr in group_results}

    # Get 8 best 3rd places (from your previous code)
    best_thirds_objs = get_best_third_places(group_results)
    thirds = [t.name for t in best_thirds_objs]

    # 2. ROUND OF 32 (Mapping from your Image)
    # We create a dictionary to store winners of Match IDs (M73, M74...)
    match_history=[]
    m = {}

    # Left Side of Bracket
    m['M74'],data = simulate_ko(winners['E'], thirds[0], model_params)  # 1E vs 3ABCDF
    match_history.append(data)
    m['M77'],data = simulate_ko(winners['I'], thirds[1], model_params)  # 1I vs 3CDFG H
    match_history.append(data)
    m['M73'],data = simulate_ko(runners['A'], runners['B'], model_params)  # 2A vs 2B
    match_history.append(data)
    m['M75'],data = simulate_ko(winners['F'], runners['C'], model_params)  # 1F vs 2C
    match_history.append(data)
    m['M83'],data = simulate_ko(runners['K'], runners['L'], model_params)  # 2K vs 2L
    match_history.append(data)
    m['M84'],data = simulate_ko(winners['H'], runners['J'], model_params)  # 1H vs 2J
    match_history.append(data)
    m['M81'],data = simulate_ko(winners['D'], thirds[2], model_params)  # 1D vs 3BEFIJ
    match_history.append(data)
    m['M82'],data = simulate_ko(winners['G'], thirds[3], model_params)  # 1G vs 3AEHIJ
    match_history.append(data)

    # Right Side of Bracket
    m['M76'],data  = simulate_ko(winners['C'], runners['F'], model_params)  # 1C vs 2F
    match_history.append(data)
    m['M78'] ,data = simulate_ko(runners['E'], runners['I'], model_params)  # 2E vs 2I
    match_history.append(data)
    m['M79'] ,data = simulate_ko(winners['A'], thirds[4], model_params)  # 1A vs 3CEFHI
    match_history.append(data)
    m['M80'] ,data = simulate_ko(winners['L'], thirds[5], model_params)  # 1L vs 3EHIJK
    match_history.append(data)
    m['M86'] ,data = simulate_ko(winners['J'], runners['H'], model_params)  # 1J vs 2H
    match_history.append(data)
    m['M88'],data  = simulate_ko(runners['D'], runners['G'], model_params)  # 2D vs 2G
    match_history.append(data)
    m['M85'] ,data = simulate_ko(runners['B'], thirds[6], model_params)  # 2B vs 3EFGIJ
    match_history.append(data)
    m['M87'] ,data = simulate_ko(winners['K'], thirds[7], model_params)  # 1K vs 3DEIJL
    match_history.append(data)

    # 3. ROUND OF 16 (Winners of M73-M88)
    m['M89'] ,data = simulate_ko(m['M74'], m['M77'], model_params)
    match_history.append(data)
    m['M90'] ,data = simulate_ko(m['M73'], m['M75'], model_params)
    match_history.append(data)
    m['M93'],data  = simulate_ko(m['M83'], m['M84'], model_params)
    match_history.append(data)
    m['M94'] ,data = simulate_ko(m['M81'], m['M82'], model_params)
    match_history.append(data)

    m['M91'] ,data = simulate_ko(m['M76'], m['M78'], model_params)
    match_history.append(data)
    m['M92'] ,data = simulate_ko(m['M79'], m['M80'], model_params)
    match_history.append(data)
    m['M95'] ,data = simulate_ko(m['M86'], m['M88'], model_params)
    match_history.append(data)
    m['M96'] ,data = simulate_ko(m['M85'], m['M87'], model_params)
    match_history.append(data)

    # 4. QUARTER FINALS
    m['M97'] ,data = simulate_ko(m['M89'], m['M90'], model_params)
    match_history.append(data)
    m['M98'] ,data = simulate_ko(m['M93'], m['M94'], model_params)
    match_history.append(data)
    m['M99'] ,data = simulate_ko(m['M91'], m['M92'], model_params)
    match_history.append(data)
    m['M100'] ,data = simulate_ko(m['M95'], m['M96'], model_params)
    match_history.append(data)

    # 5. SEMI FINALS
    m['M101'] ,data = simulate_ko(m['M97'], m['M98'], model_params)
    match_history.append(data)
    m['M102'] ,data = simulate_ko(m['M99'], m['M100'], model_params)
    match_history.append(data)

    # 6. FINAL
    champion ,data  = simulate_ko(m['M101'], m['M102'], model_params)
    match_history.append(data)

    result={
        "Winner": champion,
        "Finalist": [m['M101'], m['M102']],
        "SemiFinalists": [m['M97'], m['M98'], m['M99'], m['M100']],
        "QuarterFinalists": [m['M89'], m['M90'], m['M91'], m['M92'], m['M93'], m['M94'], m['M95'], m['M96']]
    }


    return champion,result,match_history


def simulate_ko(team_a: str, team_b: str, params: dict) -> Tuple[str, dict]:
    """
    Returns (WinnerName, MatchDataDictionary)
    """
    # Assuming simulate_match returns an object with .home_goals, .away_goals, etc.
    res = simulate_match(team_a, team_b, params, neutral=True)

    winner = res.winner
    if winner == "Draw":
        # Handle penalties
        winner = random.choice([team_a, team_b])
        res.was_penalties = True
    else:
        res.was_penalties = False

    # Return the winner name (for bracket flow) and the data (for history)
    match_data = {
        "winner": winner,
        "home": team_a,
        "away": team_b,
        "home_goals": res.home_goals,
        "away_goals": res.away_goals,
        "penalties": res.was_penalties

    }
    return winner, match_data