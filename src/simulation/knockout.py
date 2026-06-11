# src/simulation/knockout.py

from typing import List, Dict, Tuple
import random
from src.simulation.models import GroupResult, TeamStats, MatchResult
from src.simulation.match_simulator import simulate_match, get_match_electric_factor


def get_best_third_places(group_results: List[GroupResult]) -> List[TeamStats]:
    all_thirds = []
    for grp in group_results:
        all_thirds.append(grp.standings[2])
    all_thirds.sort(key=lambda x: (x.points, x.goal_difference, x.goals_for), reverse=True)
    return all_thirds[:8]


def get_loser(match_data: dict) -> str:
    winner = match_data["winner"]
    if winner == match_data["home"]: 
        return match_data["away"]
    else: 
        return match_data["home"]


def simulate_ko(team_a: str, team_b: str, params: dict, electric: str) -> Tuple[str, dict]:
    match_params = params.copy()
    match_params['electric_factor'] = get_match_electric_factor(electric)
    # CORREGIDO: Ahora sí pasamos 'match_params' para activar la Binomial Negativa
    res = simulate_match(team_a, team_b, match_params, neutral=True)

    winner = res.winner
    if winner == "Draw":
        winner = random.choice([team_a, team_b])
        res.was_penalties = True
    else:
        res.was_penalties = False

    match_data = {
        "match_id": "", # Se llenará dinámicamente en run_tournament_knockout
        "winner": winner,
        "home": team_a,
        "away": team_b,
        "home_goals": res.home_goals,
        "away_goals": res.away_goals,
        "penalties": res.was_penalties,
        "stage": electric
    }
    return winner, match_data


def run_tournament_knockout(group_results: List[GroupResult], model_params: dict):
    winners = {gr.group_name: gr.standings[0].name for gr in group_results}
    runners = {gr.group_name: gr.standings[1].name for gr in group_results}

    best_thirds_objs = get_best_third_places(group_results)
    thirds_teams = [t.name for t in best_thirds_objs]

    team_to_group = {t.name: gr.group_name for gr in group_results for t in gr.standings}
    hosts = ['E', 'I', 'D', 'G', 'A', 'L', 'B', 'K']

    def find_matching(idx, current):
        if idx == len(thirds_teams):
            return current
        t_name = thirds_teams[idx]
        t_group = team_to_group[t_name]
        for h in hosts:
            if h not in current and h != t_group:
                new_curr = current.copy()
                new_curr[h] = t_name
                res = find_matching(idx + 1, new_curr)
                if res:
                    return res
        return None

    matched_thirds = find_matching(0, {})
    if not matched_thirds:
        matched_thirds = {hosts[i]: thirds_teams[i] for i in range(8)}

    match_history = []
    m = {}

    # Helper para simular y guardar ID
    def step_ko(m_id, team_h, team_a, stage):
        winner, data = simulate_ko(team_h, team_a, model_params, stage)
        data["match_id"] = m_id
        match_history.append(data)
        return winner

    # 2. ROUND OF 32 (16 partidos)
    m['M74'] = step_ko('M74', winners['E'], matched_thirds['E'], "round_of_32")
    m['M77'] = step_ko('M77', winners['I'], matched_thirds['I'], "round_of_32")
    m['M73'] = step_ko('M73', runners['A'], runners['B'], "round_of_32")
    m['M75'] = step_ko('M75', winners['F'], runners['C'], "round_of_32")
    m['M83'] = step_ko('M83', runners['K'], runners['L'], "round_of_32")
    m['M84'] = step_ko('M84', winners['H'], runners['J'], "round_of_32")
    m['M81'] = step_ko('M81', winners['D'], matched_thirds['D'], "round_of_32")
    m['M82'] = step_ko('M82', winners['G'], matched_thirds['G'], "round_of_32")

    m['M76'] = step_ko('M76', winners['C'], runners['F'], "round_of_32")
    m['M78'] = step_ko('M78', runners['E'], runners['I'], "round_of_32")
    m['M79'] = step_ko('M79', winners['A'], matched_thirds['A'], "round_of_32")
    m['M80'] = step_ko('M80', winners['L'], matched_thirds['L'], "round_of_32")
    m['M86'] = step_ko('M86', winners['J'], runners['H'], "round_of_32")
    m['M88'] = step_ko('M88', runners['D'], runners['G'], "round_of_32")
    m['M85'] = step_ko('M85', winners['B'], matched_thirds['B'], "round_of_32")
    m['M87'] = step_ko('M87', winners['K'], matched_thirds['K'], "round_of_32")

    # Guardar equipos que jugaron 32avos
    r32_teams = list(winners.values()) + list(runners.values()) + thirds_teams

    # 3. ROUND OF 16 (Octavos de Final)
    m['M89'] = step_ko('M89', m['M74'], m['M77'], "round_of_16")
    m['M90'] = step_ko('M90', m['M73'], m['M75'], "round_of_16")
    m['M93'] = step_ko('M93', m['M83'], m['M84'], "round_of_16")
    m['M94'] = step_ko('M94', m['M81'], m['M82'], "round_of_16")

    m['M91'] = step_ko('M91', m['M76'], m['M78'], "round_of_16")
    m['M92'] = step_ko('M92', m['M79'], m['M80'], "round_of_16")
    m['M95'] = step_ko('M95', m['M86'], m['M88'], "round_of_16")
    m['M96'] = step_ko('M96', m['M85'], m['M87'], "round_of_16")

    r16_teams = [m['M74'], m['M77'], m['M73'], m['M75'], m['M83'], m['M84'], m['M81'], m['M82'],
                 m['M76'], m['M78'], m['M79'], m['M80'], m['M86'], m['M88'], m['M85'], m['M87']]

    # 4. QUARTER FINALS
    m['M97'] = step_ko('M97', m['M89'], m['M90'], "quarter_finals")
    m['M98'] = step_ko('M98', m['M93'], m['M94'], "quarter_finals")
    m['M99'] = step_ko('M99', m['M91'], m['M92'], "quarter_finals")
    m['M100'] = step_ko('M100', m['M95'], m['M96'], "quarter_finals")

    qf_teams = [m['M89'], m['M90'], m['M91'], m['M92'], m['M93'], m['M94'], m['M95'], m['M96']]

    # 5. SEMI FINALS
    m['M101'] = step_ko('M101', m['M97'], m['M98'], "semi_finals")
    m['M102'] = step_ko('M102', m['M99'], m['M100'], "semi_finals")

    sf_teams = [m['M97'], m['M98'], m['M99'], m['M100']]

    # 6. THIRD PLACE
    # CORREGIDO: M101 es penúltimo (-2 en historial de semifinales) y M102 es el último (-1)
    m['M103'] = step_ko('M103', get_loser(match_history[-2]), get_loser(match_history[-1]), "third_place")

    # 7. FINAL
    champion = step_ko('M104', m['M101'], m['M102'], "final")

    result = {
        "Winner": champion,
        "Finalist": [m['M101'], m['M102']],
        "SemiFinalists": sf_teams,
        "QuarterFinalists": qf_teams,
        "RoundOf16": r16_teams,
        "RoundOf32": r32_teams
    }

    return champion, result, match_history