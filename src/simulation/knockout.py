from typing import List, Dict, Tuple
import random
from src.simulation.models import GroupResult, TeamStats, MatchResult
from src.simulation.match_simulator import simulate_match, get_match_electric_factor


def get_best_third_places(group_results: List[GroupResult]) -> List[TeamStats]:
    """
    Takes the results from all 12 groups and returns the 8 best 3rd-place teams.
    """
    all_thirds = []
    for grp in group_results:
        # Assuming standings are already sorted, the 3rd team is at index 2
        all_thirds.append(grp.standings[2])

    # Sort all 3rd place teams by Points, then GD, then GF
    all_thirds.sort(key=lambda x: (x.points, x.goal_difference, x.goals_for), reverse=True)

    return all_thirds[:8]


def run_tournament_knockout(group_results: List[GroupResult], model_params: dict):
    # 1. Identify all qualified teams
    winners = {gr.group_name: gr.standings[0].name for gr in group_results}
    runners = {gr.group_name: gr.standings[1].name for gr in group_results}

    # Get 8 best 3rd places
    best_thirds_objs = get_best_third_places(group_results)
    thirds_teams = [t.name for t in best_thirds_objs]

    # Diccionario para mapear cada equipo a su grupo original
    # Esto es vital para saber de qué grupo viene cada 3ero y evitar repeticiones
    team_to_group = {t.name: gr.group_name for gr in group_results for t in gr.standings}

    # Estos son los 8 ganadores de grupo que según el bracket juegan contra terceros
    hosts = ['E', 'I', 'D', 'G', 'A', 'L', 'B', 'K']

    # Algoritmo de Backtracking para asignar dinámicamente los 3eros a los ganadores de grupo.
    # Garantiza que un tercero nunca juegue contra el ganador de su mismo grupo.
    def find_matching(idx, current):
        if idx == len(thirds_teams):
            return current

        t_name = thirds_teams[idx]
        t_group = team_to_group[t_name]

        for h in hosts:
            # Condición FIFA: El grupo del ganador ('h') no puede ser igual al grupo del tercero ('t_group')
            if h not in current and h != t_group:
                new_curr = current.copy()
                new_curr[h] = t_name
                res = find_matching(idx + 1, new_curr)
                if res:
                    return res
        return None

    matched_thirds = find_matching(0, {})

    # Failsafe por si llegara a ocurrir un caso extremo (matemáticamente evitado por la estructura)
    if not matched_thirds:
        matched_thirds = {hosts[i]: thirds_teams[i] for i in range(8)}

    # 2. ROUND OF 32
    match_history = []
    m = {}

    # Left Side of Bracket
    m['M74'], data = simulate_ko(winners['E'], matched_thirds['E'], model_params, "round_of_32")
    match_history.append(data)
    m['M77'], data = simulate_ko(winners['I'], matched_thirds['I'], model_params, "round_of_32")
    match_history.append(data)
    m['M73'], data = simulate_ko(runners['A'], runners['B'], model_params, "round_of_32")  # 2A vs 2B
    match_history.append(data)
    m['M75'], data = simulate_ko(winners['F'], runners['C'], model_params, "round_of_32")  # 1F vs 2C
    match_history.append(data)
    m['M83'], data = simulate_ko(runners['K'], runners['L'], model_params, "round_of_32")  # 2K vs 2L
    match_history.append(data)
    m['M84'], data = simulate_ko(winners['H'], runners['J'], model_params, "round_of_32")  # 1H vs 2J
    match_history.append(data)
    m['M81'], data = simulate_ko(winners['D'], matched_thirds['D'], model_params, "round_of_32")
    match_history.append(data)
    m['M82'], data = simulate_ko(winners['G'], matched_thirds['G'], model_params, "round_of_32")

    # Right Side of Bracket
    m['M76'], data = simulate_ko(winners['C'], runners['F'], model_params, "round_of_32")  # 1C vs 2F
    match_history.append(data)
    m['M78'], data = simulate_ko(runners['E'], runners['I'], model_params, "round_of_32")  # 2E vs 2I
    match_history.append(data)
    m['M79'], data = simulate_ko(winners['A'], matched_thirds['A'], model_params, "round_of_32")
    match_history.append(data)
    m['M80'], data = simulate_ko(winners['L'], matched_thirds['L'], model_params, "round_of_32")
    match_history.append(data)
    m['M86'], data = simulate_ko(winners['J'], runners['H'], model_params, "round_of_32")  # 1J vs 2H
    match_history.append(data)
    m['M88'], data = simulate_ko(runners['D'], runners['G'], model_params, "round_of_32")  # 2D vs 2G
    match_history.append(data)
    # CORRECCIÓN BUG M85: Originalmente tenías runners['B'] (quien ya jugaba en el M73). Es el turno de winners['B']
    m['M85'], data = simulate_ko(winners['B'], matched_thirds['B'], model_params, "round_of_32")
    match_history.append(data)
    m['M87'], data = simulate_ko(winners['K'], matched_thirds['K'], model_params, "round_of_32")
    match_history.append(data)

    # 3. ROUND OF 16 (Winners of M73-M88)
    m['M89'], data = simulate_ko(m['M74'], m['M77'], model_params, "round_of_16")
    match_history.append(data)
    m['M90'], data = simulate_ko(m['M73'], m['M75'], model_params, "round_of_16")
    match_history.append(data)
    m['M93'], data = simulate_ko(m['M83'], m['M84'], model_params, "round_of_16")
    match_history.append(data)
    m['M94'], data = simulate_ko(m['M81'], m['M82'], model_params, "round_of_16")
    match_history.append(data)

    m['M91'], data = simulate_ko(m['M76'], m['M78'], model_params, "round_of_16")
    match_history.append(data)
    m['M92'], data = simulate_ko(m['M79'], m['M80'], model_params, "round_of_16")
    match_history.append(data)
    m['M95'], data = simulate_ko(m['M86'], m['M88'], model_params, "round_of_16")
    match_history.append(data)
    m['M96'], data = simulate_ko(m['M85'], m['M87'], model_params, "round_of_16")
    match_history.append(data)

    # 4. QUARTER FINALS
    m['M97'], data = simulate_ko(m['M89'], m['M90'], model_params, "quarter_finals")
    match_history.append(data)
    m['M98'], data = simulate_ko(m['M93'], m['M94'], model_params, "quarter_finals")
    match_history.append(data)
    m['M99'], data = simulate_ko(m['M91'], m['M92'], model_params, "quarter_finals")
    match_history.append(data)
    m['M100'], data = simulate_ko(m['M95'], m['M96'], model_params, "quarter_finals")
    match_history.append(data)

    # 5. SEMI FINALS
    m['M101'], data = simulate_ko(m['M97'], m['M98'], model_params, "semi_finals")
    match_history.append(data)
    m['M102'], data = simulate_ko(m['M99'], m['M100'], model_params, "semi_finals")
    match_history.append(data)

    #Third Place
    m['M103'], data = simulate_ko(get_loser(match_history[-1]), get_loser(match_history[-1]), model_params, "third_place")
    match_history.append(data)

    # 6. FINAL
    champion, data = simulate_ko(m['M101'], m['M102'], model_params, "final")
    match_history.append(data)

    result = {
        "Winner": champion,
        "Finalist": [m['M101'], m['M102']],
        "SemiFinalists": [m['M97'], m['M98'], m['M99'], m['M100']],
        "QuarterFinalists": [m['M89'], m['M90'], m['M91'], m['M92'], m['M93'], m['M94'], m['M95'], m['M96']]
    }

    return champion, result, match_history



def get_loser(match_data: dict) -> str:
    winner = match_data["winner"]
    if winner == match_data["home"]: return match_data["away"]
    else: return match_data["home"]

def simulate_ko(team_a: str, team_b: str, params: dict,electric: str) -> Tuple[str, dict]:
    """
    Returns (WinnerName, MatchDataDictionary)
    """
    match_params = params.copy()
    match_params['electric_factor']=get_match_electric_factor(electric)
    res = simulate_match(team_a, team_b, params, neutral=True)

    winner = res.winner
    if winner == "Draw":
        # Handle penalties
        winner = random.choice([team_a, team_b])
        res.was_penalties = True
    else:
        res.was_penalties = False

    match_data = {
        "winner": winner,
        "home": team_a,
        "away": team_b,
        "home_goals": res.home_goals,
        "away_goals": res.away_goals,
        "penalties": res.was_penalties
    }
    return winner, match_data