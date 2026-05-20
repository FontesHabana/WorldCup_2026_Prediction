from itertools import combinations
from src.simulation.models import TeamStats, MatchResult, GroupResult
from src.simulation.match_simulator import simulate_match


def simulate_group_stage(groups_config, model_params):
    """
    Simula toda la fase de grupos.

    groups_config: dict {'A': ['Argentina', 'Mexico', ...], 'B': [...]}
    model_params: dict con los parámetros attack, defense, gamma, rho del fit.
    """
    all_group_results = []

    for group_name, team_names in groups_config.items():
        # 1. Inicializar stats de cada equipo
        standings = {name: TeamStats(name) for name in team_names}
        group_matches = []

        # 2. Generar todos los partidos del grupo (6 partidos por grupo de 4)
        for home_name, away_name in combinations(team_names, 2):
            # Simular el partido individual
            # Nota: En el mundial casi todos son neutrales, pasamos neutral=True
            res = simulate_match(home_name, away_name, model_params, neutral=True)
            group_matches.append(res)

            # 3. Actualizar estadísticas de los equipos
            h_stats = standings[home_name]
            a_stats = standings[away_name]

            h_stats.goals_for += res.home_goals
            h_stats.goals_against += res.away_goals
            a_stats.goals_for += res.away_goals
            a_stats.goals_against += res.home_goals

            if res.winner == home_name:
                h_stats.points += 3
            elif res.winner == away_name:
                a_stats.points += 3
            else:
                h_stats.points += 1
                a_stats.points += 1

        # 4. Crear el objeto GroupResult y ordenar posiciones
        result = GroupResult(
            group_name=group_name,
            standings=list(standings.values()),
            matches=group_matches
        )

        # Ordenar por puntos, luego diferencia de goles, luego goles a favor
        result.sort_standings()

        all_group_results.append(result)

    return all_group_results