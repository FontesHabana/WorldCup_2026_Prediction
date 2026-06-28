# run_direct_knockout_simulation.py

import os
import joblib
import pandas as pd
from collections import Counter, defaultdict
from tqdm import tqdm

from src.simulation.knockout import simulate_ko
from src.utils.persistence import load_model

# =====================================================================
# CONFIGURACIÓN MANUAL DE LOS CRUCES (RONDA DE 32)
# Organizados consecutivamente. Los partidos contiguos se enfrentan en la siguiente ronda.
# =====================================================================
INITIAL_MATCHUPS = {
    "M73": ("germany", "paraguay"),
    "M74": ("france", "sweden"),
    "M75": ("south africa", "canada"),
    "M76": ("netherlands", "morocco"),
    "M77": ("portugal", "croatia"),
    "M78": ("spain", "austria"),
    "M79": ("united states", "bosnia and herzegovina"),
    "M80": ("belgium", "senegal"),

    "M81": ("brazil", "japan"),
    "M82": ("ivory coast", "norway"),
    "M83": ("mexico", "ecuador"),
    "M84": ("england", "dr congo"),
    "M85": ("argentina", "cape verde"),
    "M86": ("australia", "egypt"),
    "M87": ("switzerland", "algeria"),
    "M88": ("colombia", "ghana")
}

# Mapeo descriptivo para etapas del torneo
STAGE_MAP = {
    'M73': 'Round of 32', 'M74': 'Round of 32', 'M75': 'Round of 32', 'M76': 'Round of 32',
    'M77': 'Round of 32', 'M78': 'Round of 32', 'M79': 'Round of 32', 'M80': 'Round of 32',
    'M81': 'Round of 32', 'M82': 'Round of 32', 'M83': 'Round of 32', 'M84': 'Round of 32',
    'M85': 'Round of 32', 'M86': 'Round of 32', 'M87': 'Round of 32', 'M88': 'Round of 32',
    'M89': 'Round of 16', 'M90': 'Round of 16', 'M91': 'Round of 16', 'M92': 'Round of 16',
    'M93': 'Round of 16', 'M94': 'Round of 16', 'M95': 'Round of 16', 'M96': 'Round of 16',
    'M97': 'Quarter-Finals', 'M98': 'Quarter-Finals', 'M99': 'Quarter-Finals', 'M100': 'Quarter-Finals',
    'M101': 'Semi-Finals', 'M102': 'Semi-Finals',
    'M103': 'Third Place Playoff',
    'M104': 'Final'
}


def is_valid_team(name: str) -> bool:
    """Valida si la cadena representa una selección participante real."""
    return name is not None and name.strip() != "" and name.strip() != "-"


def run_direct_knockout(r32_matchups: dict, model_params: dict):
    """Ejecuta el cuadro eliminatorio completo de forma consecutiva 2 a 2."""
    match_history = []
    m = {}

    def step_ko(m_id, team_h, team_a, stage):
        valid_h = is_valid_team(team_h)
        valid_a = is_valid_team(team_a)

        # Caso 1: Ambos espacios están vacíos o son inválidos
        if not valid_h and not valid_a:
            winner = "-"
            data = {
                "match_id": m_id, "winner": "-", "home": "-", "away": "-",
                "home_goals": 0, "away_goals": 0, "penalties": False, "stage": stage
            }
            match_history.append(data)
            return winner

        # Caso 2: Solo el equipo local es inválido (el visitante avanza por walkover)
        elif not valid_h:
            winner = team_a
            data = {
                "match_id": m_id, "winner": team_a, "home": "-", "away": team_a,
                "home_goals": 0, "away_goals": 3, "penalties": False, "stage": stage
            }
            match_history.append(data)
            return winner

        # Caso 3: Solo el equipo visitante es inválido (el local avanza por walkover)
        elif not valid_a:
            winner = team_h
            data = {
                "match_id": m_id, "winner": team_h, "home": team_h, "away": "-",
                "home_goals": 3, "away_goals": 0, "penalties": False, "stage": stage
            }
            match_history.append(data)
            return winner

        # Caso 4: Ambos equipos son válidos (se simula el partido ordinariamente)
        else:
            winner, data = simulate_ko(team_h, team_a, model_params, stage)
            data["match_id"] = m_id
            match_history.append(data)
            return winner

    # 1. RONDA DE 32 (16 partidos iniciales)
    for m_id in [f'M{i}' for i in range(73, 89)]:
        team_h, team_a = r32_matchups[m_id]
        m[m_id] = step_ko(m_id, team_h, team_a, "round_of_32")

    r32_teams = []
    for pair in r32_matchups.values():
        r32_teams.extend(list(pair))

    # 2. OCTAVOS DE FINAL (Consecutivos 2 a 2 de la ronda anterior)
    m['M89'] = step_ko('M89', m['M73'], m['M74'], "round_of_16")  # Ganador M73 vs Ganador M74
    m['M90'] = step_ko('M90', m['M75'], m['M76'], "round_of_16")  # Ganador M75 vs Ganador M76
    m['M91'] = step_ko('M91', m['M77'], m['M78'], "round_of_16")  # Ganador M77 vs Ganador M78
    m['M92'] = step_ko('M92', m['M79'], m['M80'], "round_of_16")  # Ganador M79 vs Ganador M80

    m['M93'] = step_ko('M93', m['M81'], m['M82'], "round_of_16")  # Ganador M81 vs Ganador M82
    m['M94'] = step_ko('M94', m['M83'], m['M84'], "round_of_16")  # Ganador M83 vs Ganador M84
    m['M95'] = step_ko('M95', m['M85'], m['M86'], "round_of_16")  # Ganador M85 vs Ganador M86
    m['M96'] = step_ko('M96', m['M87'], m['M88'], "round_of_16")  # Ganador M87 vs Ganador M88

    # Clasificados a octavos de final (los ganadores de los 16 partidos de 32avos)
    r16_teams = [m[f'M{i}'] for i in range(73, 89)]

    # 3. CUARTOS DE FINAL (Consecutivos 2 a 2 de octavos de final)
    m['M97'] = step_ko('M97', m['M89'], m['M90'], "quarter_finals")  # Superior Izquierda (M73-M76)
    m['M98'] = step_ko('M98', m['M91'], m['M92'], "quarter_finals")  # Inferior Izquierda (M77-M80)
    m['M99'] = step_ko('M99', m['M93'], m['M94'], "quarter_finals")  # Superior Derecha   (M81-M84)
    m['M100'] = step_ko('M100', m['M95'], m['M96'], "quarter_finals")  # Inferior Derecha   (M85-M88)

    qf_teams = [m['M89'], m['M90'], m['M91'], m['M92'], m['M93'], m['M94'], m['M95'], m['M96']]

    # 4. SEMIFINALES
    m['M101'] = step_ko('M101', m['M97'], m['M98'], "semi_finals")  # Lado Izquierdo del cuadro
    m['M102'] = step_ko('M102', m['M99'], m['M100'], "semi_finals")  # Lado Derecho del cuadro

    sf_teams = [m['M97'], m['M98'], m['M99'], m['M100']]

    def get_loser(match_data: dict) -> str:
        winner = match_data["winner"]
        return match_data["away"] if winner == match_data["home"] else match_data["home"]

    # 5. TERCER LUGAR
    m['M103'] = step_ko('M103', get_loser(match_history[-2]), get_loser(match_history[-1]), "third_place")

    # 6. GRAN FINAL
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


class DirectKnockoutAnalyzer:
    def __init__(self):
        self.progression_counts = {
            "RoundOf32": Counter(),
            "RoundOf16": Counter(),
            "QuarterFinalists": Counter(),
            "SemiFinalists": Counter(),
            "Finalist": Counter(),
            "Winner": Counter()
        }
        self.matchup_counts = defaultdict(Counter)
        self.matchup_winners = defaultdict(lambda: defaultdict(Counter))
        self.matchup_scores = defaultdict(lambda: defaultdict(Counter))
        self.matchup_penalties = defaultdict(lambda: defaultdict(Counter))
        self.matchup_penalty_winners = defaultdict(lambda: defaultdict(Counter))
        self.total_simulations = 0

    def aggregate_iteration(self, knockout_result, match_history):
        self.total_simulations += 1

        for stage, teams_list in knockout_result.items():
            if stage == "Winner":
                self.progression_counts["Winner"][teams_list] += 1
            else:
                for team in teams_list:
                    self.progression_counts[stage][team] += 1

        for match in match_history:
            m_id = match["match_id"]
            team_h = match["home"]
            team_a = match["away"]
            sorted_teams = tuple(sorted([team_h, team_a]))

            if team_h == sorted_teams[0]:
                goals_0, goals_1 = match["home_goals"], match["away_goals"]
            else:
                goals_0, goals_1 = match["away_goals"], match["home_goals"]

            winner = match["winner"]
            was_penalties = match.get("penalties", False)

            self.matchup_counts[m_id][sorted_teams] += 1
            self.matchup_winners[m_id][sorted_teams][winner] += 1
            self.matchup_scores[m_id][sorted_teams][(goals_0, goals_1)] += 1
            self.matchup_penalties[m_id][sorted_teams][was_penalties] += 1

            if was_penalties:
                self.matchup_penalty_winners[m_id][sorted_teams][winner] += 1

    def _get_sequential_matchups(self) -> dict:
        """
        Determina secuencialmente los emparejamientos y ganadores lógicos del bracket.
        Esto previene la discrepancia entre el ganador predicho de una ronda y los
        participantes de la siguiente fase en el reporte.
        """
        winner_map = {}
        active_matchups = {}

        # Mapeo de predecesores para construir el árbol secuencial
        parent_map = {
            # Ronda de 16
            'M89': ('M73', 'M74'),
            'M90': ('M75', 'M76'),
            'M91': ('M77', 'M78'),
            'M92': ('M79', 'M80'),
            'M93': ('M81', 'M82'),
            'M94': ('M83', 'M84'),
            'M95': ('M85', 'M86'),
            'M96': ('M87', 'M88'),
            # Cuartos de final
            'M97': ('M89', 'M90'),
            'M98': ('M91', 'M92'),
            'M99': ('M93', 'M94'),
            'M100': ('M95', 'M96'),
            # Semifinales
            'M101': ('M97', 'M98'),
            'M102': ('M99', 'M100'),
            # Final
            'M104': ('M101', 'M102')
        }

        def get_sort_key(x):
            num_part = "".join(filter(str.isdigit, str(x)))
            return int(num_part) if num_part else 999

        all_m_ids = sorted(self.matchup_counts.keys(), key=get_sort_key)

        for m_id in all_m_ids:
            if m_id == 'M103':
                # El partido de tercer lugar se juega entre los perdedores de las semifinales M101 y M102
                if 'M101' in active_matchups and 'M102' in active_matchups:
                    m101_match = active_matchups['M101']
                    w101 = winner_map['M101']
                    l101 = m101_match[1] if m101_match[0] == w101 else m101_match[0]

                    m102_match = active_matchups['M102']
                    w102 = winner_map['M102']
                    l102 = m102_match[1] if m102_match[0] == w102 else m102_match[0]

                    matchup = tuple(sorted([l101, l102]))
                else:
                    matchup, _ = self.matchup_counts[m_id].most_common(1)[0]
            elif m_id in parent_map:
                p1, p2 = parent_map[m_id]
                t1 = winner_map.get(p1, "-")
                t2 = winner_map.get(p2, "-")
                matchup = tuple(sorted([t1, t2]))
            else:
                # Ronda de 32 (M73-M88) - Emparejamientos fijos iniciales
                if m_id in self.matchup_counts and self.matchup_counts[m_id]:
                    matchup, _ = self.matchup_counts[m_id].most_common(1)[0]
                else:
                    continue

            # Validar si este enfrentamiento específico ocurrió en la simulación
            freq = self.matchup_counts[m_id].get(matchup, 0)
            if freq == 0 and self.matchup_counts[m_id]:
                # Fallback preventivo si no existieran registros bajo esa combinación exacta
                matchup, freq = self.matchup_counts[m_id].most_common(1)[0]

            # Obtener el ganador más común para este enfrentamiento específico
            winner_counter = self.matchup_winners[m_id][matchup]
            if winner_counter:
                winner, _ = winner_counter.most_common(1)[0]
            else:
                winner = "-"

            winner_map[m_id] = winner
            active_matchups[m_id] = matchup

        return active_matchups

    def generate_progression_report(self) -> pd.DataFrame:
        all_teams = set()
        for stage in self.progression_counts.values():
            all_teams.update(stage.keys())

        # Remover indicadores genéricos de los reportes para mantener limpieza de datos
        if "-" in all_teams:
            all_teams.remove("-")
        if "" in all_teams:
            all_teams.remove("")

        report_data = []
        for team in all_teams:
            row = {"Team": team}
            for stage, counter in self.progression_counts.items():
                prob = (counter[team] / self.total_simulations) * 100
                row[stage] = round(prob, 2)
            report_data.append(row)

        df = pd.DataFrame(report_data)
        rename_dict = {
            "RoundOf32": "Round-of-32 (%)",
            "RoundOf16": "Round-of-16 (%)",
            "QuarterFinalists": "Quarter-Final (%)",
            "SemiFinalists": "Semi-Final (%)",
            "Finalist": "Finalist (%)",
            "Winner": "Champion (%)"
        }
        df = df.rename(columns=rename_dict)
        stages_order = ["Round-of-32 (%)", "Round-of-16 (%)", "Quarter-Final (%)", "Semi-Final (%)", "Finalist (%)",
                        "Champion (%)"]

        for col in stages_order:
            if col not in df.columns:
                df[col] = 0.0

        df = df.set_index("Team")
        df = df[stages_order]
        return df.sort_values(by="Champion (%)", ascending=False)

    def generate_bracket_report(self) -> pd.DataFrame:
        report_rows = []
        active_matchups = self._get_sequential_matchups()

        def get_sort_key(x):
            num_part = "".join(filter(str.isdigit, str(x)))
            return int(num_part) if num_part else 999

        for m_id in sorted(self.matchup_counts.keys(), key=get_sort_key):
            if m_id not in active_matchups:
                continue

            matchup = active_matchups[m_id]
            matchup_freq = self.matchup_counts[m_id].get(matchup, 0)

            # Fallback en caso de frecuencia nula para mantener estabilidad
            if matchup_freq == 0:
                matchup, matchup_freq = self.matchup_counts[m_id].most_common(1)[0]

            matchup_prob = (matchup_freq / self.total_simulations) * 100

            score_counter = self.matchup_scores[m_id][matchup]
            if score_counter:
                most_common_score, score_freq = score_counter.most_common(1)[0]
                goals_0, goals_1 = most_common_score
                score_str = f"{matchup[0]} {goals_0} - {goals_1} {matchup[1]}"
                score_prob = (score_freq / matchup_freq) * 100
            else:
                score_str = f"{matchup[0]} 0 - 0 {matchup[1]}"
                score_prob = 0.0

            winner_counter = self.matchup_winners[m_id][matchup]
            if winner_counter:
                most_common_winner, winner_freq = winner_counter.most_common(1)[0]
                winner_prob = (winner_freq / matchup_freq) * 100
            else:
                most_common_winner = "N/A"
                winner_prob = 0.0

            penalty_counter = self.matchup_penalties[m_id][matchup]
            penalty_count = penalty_counter[True]
            penalty_prob = (penalty_count / matchup_freq) * 100

            penalty_winner_counter = self.matchup_penalty_winners[m_id][matchup]
            if penalty_winner_counter and penalty_count > 0:
                most_likely_penalty_winner, pw_freq = penalty_winner_counter.most_common(1)[0]
                pw_prob = (pw_freq / penalty_count) * 100
            else:
                most_likely_penalty_winner = "N/A"
                pw_prob = 0.0

            stage_name = STAGE_MAP.get(m_id, "Unknown Stage")

            report_rows.append({
                "Match_ID": m_id,
                "Stage": stage_name,
                "Team_A": matchup[0],
                "Team_B": matchup[1],
                "Matchup_Probability_Pct": round(matchup_prob, 2),
                "Most_Probable_Score": score_str,
                "Score_Prob_Within_Matchup_Pct": round(score_prob, 2),
                "Predicted_Winner_Overall": most_common_winner,
                "Winner_Prob_Within_Matchup_Pct": round(winner_prob, 2),
                "Penalty_Probability_Pct": round(penalty_prob, 2),
                "Most_Likely_Penalty_Winner": most_likely_penalty_winner,
                "Penalty_Winner_Confidence_Pct": round(pw_prob, 2) if most_likely_penalty_winner != "N/A" else 0.0
            })

        return pd.DataFrame(report_rows)

    def generate_goal_probabilities_report(self) -> pd.DataFrame:
        report_rows = []
        active_matchups = self._get_sequential_matchups()

        def get_sort_key(x):
            num_part = "".join(filter(str.isdigit, str(x)))
            return int(num_part) if num_part else 999

        for m_id in sorted(self.matchup_counts.keys(), key=get_sort_key):
            if m_id not in active_matchups:
                continue

            matchup = active_matchups[m_id]
            matchup_freq = self.matchup_counts[m_id].get(matchup, 0)
            if matchup_freq == 0:
                continue

            score_counter = self.matchup_scores[m_id][matchup]
            stage_name = STAGE_MAP.get(m_id, "Unknown Stage")

            for (goals_0, goals_1), freq in score_counter.most_common():
                score_prob = (freq / matchup_freq) * 100
                score_str = f"{matchup[0]} {goals_0} - {goals_1} {matchup[1]}"

                report_rows.append({
                    "Match_ID": m_id,
                    "Stage": stage_name,
                    "Matchup": f"{matchup[0]} vs {matchup[1]}",
                    "Score_Line": score_str,
                    "Probability_Pct": round(score_prob, 2)
                })

        return pd.DataFrame(report_rows)


def main():
    dc_params_path = "../../models_saved/dixon_coles_params.joblib"
    ensemble_path = "../../models_saved/ensemble_model.joblib"

    print("Cargando parámetros y modelos base...")
    try:
        params = load_model(dc_params_path)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo Dixon-Coles en {dc_params_path}.")
        return

    try:
        ensemble = joblib.load(ensemble_path)
        params['ensemble'] = ensemble
        print("Ensamble XGBoost integrado con éxito.")
    except FileNotFoundError:
        params['ensemble'] = None
        print("Advertencia: No se encontró el modelo XGBoost. Simulando con Dixon-Coles puro.")

    iterations = 100000  # Se aumenta el número de iteraciones para estabilizar las tendencias estadísticas
    analyzer = DirectKnockoutAnalyzer()

    print(f"\nIniciando simulación directa desde 32avos ({iterations} iteraciones)...")
    for _ in tqdm(range(iterations), desc="Procesando eliminatorias"):
        _, ko_result, match_history = run_direct_knockout(INITIAL_MATCHUPS, params)
        analyzer.aggregate_iteration(ko_result, match_history)

    print("\nGenerando reportes...")
    df_progression = analyzer.generate_progression_report()
    df_bracket = analyzer.generate_bracket_report()
    df_goal_probs = analyzer.generate_goal_probabilities_report()

    output_dir = "../../simulation_results"
    os.makedirs(output_dir, exist_ok=True)

    progression_file = os.path.join(output_dir, "direct_knockout_progression.csv")
    bracket_file = os.path.join(output_dir, "direct_knockout_bracket_details.csv")
    goals_file = os.path.join(output_dir, "direct_knockout_goal_probabilities.csv")

    df_progression.to_csv(progression_file, index=True)
    df_bracket.to_csv(bracket_file, index=False)
    df_goal_probs.to_csv(goals_file, index=False)

    print(f"\nProceso completado. Datos exportados en la carpeta '{output_dir}':")
    print(f"  * Progresión de fases: '{progression_file}'")
    print(f"  * Detalle de bracket y penales: '{bracket_file}'")
    print(f"  * Probabilidades de goles por marcador: '{goals_file}'")


if __name__ == "__main__":
    main()