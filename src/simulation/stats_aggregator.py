import json
from collections import Counter, defaultdict
import pandas as pd


class TournamentAggregator:
    def __init__(self):
        # 1. Progression Counts
        self.reach_counts = {
            "RoundOf32": Counter(),
            "RoundOf16": Counter(),
            "QuarterFinalists": Counter(),
            "SemiFinalists": Counter(),
            "Finalist": Counter(),
            "Winner": Counter()
        }

        # 2. Group Standings Counts: {team: {pos_1: count, pos_2: count...}}
        self.group_standings = defaultdict(lambda: Counter())

        # 3. Knockout Matchup Tracker: {match_id: {(team_a, team_b): count}}
        self.matchup_counts = defaultdict(Counter)
        # Tracker de ganadores por cruce: {match_id: {winner: count}}
        self.match_winner_counts = defaultdict(Counter)

        # 4. Detalle de partidos de Fase de Grupos
        self.group_matchups = defaultdict(lambda: {
            "count": 0,
            "team_a_goals": 0,
            "team_b_goals": 0,
            "scores": Counter(),
            "team_a_wins": 0,
            "team_b_wins": 0,
            "draws": 0
        })

        # Mapeo de equipos a sus respectivos grupos: {nombre_equipo: nombre_grupo}
        self.team_groups = {}

        # 5. Tracker de marcadores específicos de cada partido (grupos y eliminatorias)
        # {match_id: { (team_a, team_b): Counter() }}
        self.match_scores = defaultdict(lambda: defaultdict(Counter))

        self.total_simulations = 0

    def aggregate_iteration(self, group_results, knockout_result, match_history):
        self.total_simulations += 1

        # A. Procesar Fase de Grupos
        for group in group_results:
            group_name = getattr(group, "group_name", "Unknown Group")

            # Acumular posiciones finales y mapear equipos a sus grupos
            for idx, team_stats in enumerate(group.standings):
                position = idx + 1
                self.group_standings[team_stats.name][position] += 1
                self.team_groups[team_stats.name] = group_name

            # Extraer y acumular estadísticas partido a partido de la fase de grupos de manera segura
            for match in getattr(group, "matches", []):
                home = getattr(match, "home", getattr(match, "home_team", None))
                away = getattr(match, "away", getattr(match, "away_team", None))

                if not home or not away:
                    continue

                sorted_teams = tuple(sorted([home, away]))

                # Extraer goles de forma robusta
                goals_home = getattr(match, "home_goals", getattr(match, "goals_home", 0))
                goals_away = getattr(match, "away_goals", getattr(match, "goals_away", 0))

                if home == sorted_teams[0]:
                    goals_a, goals_b = goals_home, goals_away
                else:
                    goals_a, goals_b = goals_away, goals_home

                stats = self.group_matchups[sorted_teams]
                stats["count"] += 1
                stats["team_a_goals"] += goals_a
                stats["team_b_goals"] += goals_b
                stats["scores"][(goals_a, goals_b)] += 1

                winner = getattr(match, "winner", "Draw")
                if winner == sorted_teams[0]:
                    stats["team_a_wins"] += 1
                elif winner == sorted_teams[1]:
                    stats["team_b_wins"] += 1
                else:
                    stats["draws"] += 1

        # B. Acumular Progresión de Knockout
        for stage, teams_list in knockout_result.items():
            if stage == "Winner":
                self.reach_counts["Winner"][teams_list] += 1
            else:
                for team in teams_list:
                    self.reach_counts[stage][team] += 1

        # C. Acumular Frecuencia de Enfrentamientos, Resultados y Marcadores para Eliminatorias
        for match in match_history:
            m_id = match["match_id"]
            sorted_teams = tuple(sorted([match["home"], match["away"]]))
            self.matchup_counts[m_id][sorted_teams] += 1
            self.match_winner_counts[m_id][match["winner"]] += 1

            # Extraer goles para guardar historial de marcadores de knockout
            goals_home = match.get("home_goals", 0)
            goals_away = match.get("away_goals", 0)

            if match["home"] == sorted_teams[0]:
                goals_a, goals_b = goals_home, goals_away
            else:
                goals_a, goals_b = goals_away, goals_home

            self.match_scores[m_id][sorted_teams][(goals_a, goals_b)] += 1

    def generate_progression_report(self) -> pd.DataFrame:
        """Genera un DataFrame con las probabilidades porcentuales de cada fase."""
        all_teams = set()
        for stage in self.reach_counts.values():
            all_teams.update(stage.keys())

        report_data = []
        for team in all_teams:
            row = {"Team": team}
            for stage, counter in self.reach_counts.items():
                prob = (counter[team] / self.total_simulations) * 100
                row[stage] = round(prob, 2)
            report_data.append(row)

        df = pd.DataFrame(report_data)
        df = df.sort_values(by="Winner", ascending=False).reset_index(drop=True)
        return df

    def generate_group_standings_report(self) -> pd.DataFrame:
        """Retorna la probabilidad de cada equipo de finalizar en la posición 1, 2, 3 o 4 de su grupo."""
        report_data = []
        for team, positions in self.group_standings.items():
            row = {"Team": team}
            for pos in [1, 2, 3, 4]:
                prob = (positions[pos] / self.total_simulations) * 100
                row[f"Position_{pos}"] = round(prob, 2)
            report_data.append(row)
        return pd.DataFrame(report_data)

    def generate_most_probable_group_stage_report(self) -> dict:
        """
        Calcula el escenario más probable de la Fase de Grupos:
        1. Determina los resultados de cada partido según su marcador más común.
        2. Simula y calcula la tabla de posiciones resultante para cada grupo aplicando tiebreakers.
        """
        group_fixtures = defaultdict(list)

        # 1. Agrupar enfrentamientos y obtener su marcador más probable
        for (team_a, team_b), stats in self.group_matchups.items():
            group_a = self.team_groups.get(team_a)
            group_b = self.team_groups.get(team_b)
            group_name = group_a if group_a == group_b else (group_a or group_b or "Unknown Group")

            if stats["scores"]:
                most_common_score_tuple, _ = stats["scores"].most_common(1)[0]
                goals_a, goals_b = most_common_score_tuple
            else:
                goals_a, goals_b = 0, 0

            group_fixtures[group_name].append({
                "team_a": team_a,
                "team_b": team_b,
                "goals_a": goals_a,
                "goals_b": goals_b,
                "score_str": f"{team_a} {goals_a} - {goals_b} {team_b}"
            })

        report = {}

        # 2. Generar las tablas de posiciones a partir de los resultados más comunes
        for group_name in sorted(group_fixtures.keys()):
            matches = group_fixtures[group_name]

            # Inicializar estadísticas para todos los equipos registrados en este grupo
            standings = defaultdict(lambda: {"P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "GD": 0, "Pts": 0})
            for team, g_name in self.team_groups.items():
                if g_name == group_name:
                    _ = standings[team]

            # Procesar cada partido simulado de forma retrospectiva
            for m in matches:
                t_a, t_b = m["team_a"], m["team_b"]
                g_a, g_b = m["goals_a"], m["goals_b"]

                standings[t_a]["P"] += 1
                standings[t_b]["P"] += 1
                standings[t_a]["GF"] += g_a
                standings[t_a]["GA"] += g_b
                standings[t_b]["GF"] += g_b
                standings[t_b]["GA"] += g_a

                if g_a > g_b:
                    standings[t_a]["W"] += 1
                    standings[t_a]["Pts"] += 3
                    standings[t_b]["L"] += 1
                elif g_a < g_b:
                    standings[t_b]["W"] += 1
                    standings[t_b]["Pts"] += 3
                    standings[t_a]["L"] += 1
                else:
                    standings[t_a]["D"] += 1
                    standings[t_a]["Pts"] += 1
                    standings[t_b]["D"] += 1
                    standings[t_b]["Pts"] += 1

            table_rows = []
            for team, stats in standings.items():
                stats["GD"] = stats["GF"] - stats["GA"]
                table_rows.append({
                    "Team": team,
                    "P": stats["P"],
                    "W": stats["W"],
                    "D": stats["D"],
                    "L": stats["L"],
                    "GF": stats["GF"],
                    "GA": stats["GA"],
                    "GD": stats["GD"],
                    "Pts": stats["Pts"]
                })

            # Criterio estándar FIFA: Puntos, Diferencia de Goles, Goles a Favor
            table_rows.sort(key=lambda x: (x["Pts"], x["GD"], x["GF"]), reverse=True)

            for idx, row in enumerate(table_rows):
                row["Pos"] = idx + 1

            report[group_name] = {
                "matches": [m["score_str"] for m in matches],
                "table": pd.DataFrame(table_rows)[["Pos", "Team", "P", "W", "D", "L", "GF", "GA", "GD", "Pts"]]
            }

        return report

    def extract_most_probable_bracket(self) -> dict:
        """
        Reconstruye el bracket determinando el emparejamiento más recurrente,
        el marcador más común de dicho partido y el ganador más votado.
        Previene colisiones entre la Final y el partido del 3er Lugar.
        """
        bracket = {}
        all_ids = list(self.matchup_counts.keys())
        final_id = None
        third_place_id = None

        # Identificar dinámicamente las IDs para el formato del Mundial 2026 (M104/M103) y 2022 (M64/M63)
        for m_id in all_ids:
            m_id_lower = str(m_id).lower()
            if m_id_lower in ["m104", "m64", "f", "final"]:
                final_id = m_id
            elif m_id_lower in ["m103", "m63", "3rd", "third", "thirdplace"]:
                third_place_id = m_id

        if not final_id or not third_place_id:
            numbered_ids = []
            for m_id in all_ids:
                num_part = "".join(filter(str.isdigit, str(m_id)))
                if num_part:
                    numbered_ids.append((int(num_part), m_id))
            if len(numbered_ids) >= 2:
                numbered_ids.sort()
                if not third_place_id:
                    third_place_id = numbered_ids[-2][1]
                if not final_id:
                    final_id = numbered_ids[-1][1]

        final_teams = set()
        if final_id and final_id in self.matchup_counts and self.matchup_counts[final_id]:
            most_common_matchup, _ = self.matchup_counts[final_id].most_common(1)[0]
            final_teams.update(most_common_matchup)

        def get_sort_key(x):
            num_part = "".join(filter(str.isdigit, str(x)))
            return int(num_part) if num_part else 999

        for m_id in sorted(self.matchup_counts.keys(), key=get_sort_key):
            if not self.matchup_counts[m_id]:
                continue

            # Evitar que los equipos finalistas jueguen el partido por el 3er lugar
            if m_id == third_place_id and final_teams:
                valid_matchups = [
                    (matchup, freq) for matchup, freq in self.matchup_counts[m_id].items()
                    if not (matchup[0] in final_teams or matchup[1] in final_teams)
                ]
                if valid_matchups:
                    valid_matchups.sort(key=lambda x: x[1], reverse=True)
                    most_common_matchup, matchup_frequency = valid_matchups[0]
                else:
                    most_common_matchup, matchup_frequency = self.matchup_counts[m_id].most_common(1)[0]
            else:
                most_common_matchup, matchup_frequency = self.matchup_counts[m_id].most_common(1)[0]

            if self.match_winner_counts[m_id]:
                most_common_winner, winner_frequency = self.match_winner_counts[m_id].most_common(1)[0]
            else:
                most_common_winner, winner_frequency = "Unknown", 0

            # Buscar el marcador más común para este cruce específico en este ID de partido
            score_counter = self.match_scores[m_id][most_common_matchup]
            if score_counter:
                most_common_score_tuple, _ = score_counter.most_common(1)[0]
                goals_a, goals_b = most_common_score_tuple
                predicted_score = f"{most_common_matchup[0]} {goals_a} - {goals_b} {most_common_matchup[1]}"
            else:
                predicted_score = "N/A"

            bracket[m_id] = {
                "matchup": f"{most_common_matchup[0]} vs {most_common_matchup[1]}",
                "matchup_probability_pct": round((matchup_frequency / self.total_simulations) * 100, 2),
                "predicted_score": predicted_score,
                "predicted_winner": most_common_winner,
                "winner_probability_pct": round((winner_frequency / self.total_simulations) * 100, 2)
            }

        return bracket

    def print_most_probable_tournament_summary(self):
        """Imprime en consola un resumen formateado del torneo más probable (grupos y bracket)."""
        print("\n" + "=" * 60)
        print("      FAS DE GRUPOS DEL MUNDIAL - ESCENARIO MÁS PROBABLE")
        print("=" * 60)

        group_report = self.generate_most_probable_group_stage_report()
        for group_name, data in group_report.items():
            print(f"\n⚽ {group_name.upper()} ⚽")
            print("Partidos:")
            for match in data["matches"]:
                print(f"  * {match}")
            print("\nTabla de Posiciones:")
            print(data["table"].to_string(index=False))
            print("-" * 50)

    def generate_most_probable_matches_per_group_report(self) -> pd.DataFrame:
        """
        Para cada grupo, recopila todos los enfrentamientos y determina su marcador más probable.
        Calcula la probabilidad de victoria de cada equipo, de empate y cuál es el partido
        más predecible (con mayor probabilidad de marcador exacto) por grupo.
        Retorna un DataFrame de Pandas listo para ser exportado a CSV.
        """
        group_fixtures = defaultdict(list)

        for (team_a, team_b), stats in self.group_matchups.items():
            group_a = self.team_groups.get(team_a)
            group_b = self.team_groups.get(team_b)
            group_name = group_a if group_a == group_b else (group_a or group_b or "Unknown Group")

            total_matches = stats["count"]
            if total_matches == 0:
                continue

            # Inicializar contadores para los resultados globales
            win_a_count = 0
            win_b_count = 0
            draw_count = 0

            # Procesar los marcadores registrados
            if stats["scores"]:
                # Obtener el marcador más común y su frecuencia
                most_common_score, score_freq = stats["scores"].most_common(1)[0]
                score_prob_pct = round((score_freq / total_matches) * 100, 2)
                goals_a, goals_b = most_common_score

                # Calcular frecuencias de victoria y empate
                for (g_a, g_b), freq in stats["scores"].items():
                    if g_a > g_b:
                        win_a_count += freq
                    elif g_b > g_a:
                        win_b_count += freq
                    else:
                        draw_count += freq
            else:
                goals_a, goals_b = 0, 0
                score_prob_pct = 0.0

            # Calcular porcentajes finales de resultados
            win_a_pct = round((win_a_count / total_matches) * 100, 2)
            win_b_pct = round((win_b_count / total_matches) * 100, 2)
            draw_pct = round((draw_count / total_matches) * 100, 2)

            group_fixtures[group_name].append({
                "Team_A": team_a,
                "Team_B": team_b,
                "Goals_A": goals_a,
                "Goals_B": goals_b,
                "Score_Probability_Pct": score_prob_pct,
                "Win_A_Probability_Pct": win_a_pct,
                "Win_B_Probability_Pct": win_b_pct,
                "Draw_Probability_Pct": draw_pct
            })

        # Determinar el partido más probable por grupo
        final_rows = []
        for group_name, matches in sorted(group_fixtures.items()):
            if not matches:
                continue

            # Encontrar el partido con la mayor probabilidad de marcador exacto en este grupo
            highest_prob_match = max(matches, key=lambda x: x["Score_Probability_Pct"])

            for m in matches:
                is_most_probable_in_group = (
                        m["Team_A"] == highest_prob_match["Team_A"] and
                        m["Team_B"] == highest_prob_match["Team_B"]
                )
                final_rows.append({
                    "Group": group_name,
                    "Team_A": m["Team_A"],
                    "Team_B": m["Team_B"],
                    "Goals_A": m["Goals_A"],
                    "Goals_B": m["Goals_B"],
                    "Score_Probability_Pct": m["Score_Probability_Pct"],
                    "Win_A_Probability_Pct": m["Win_A_Probability_Pct"],
                    "Win_B_Probability_Pct": m["Win_B_Probability_Pct"],
                    "Draw_Probability_Pct": m["Draw_Probability_Pct"],
                    "Is_Group_Most_Probable": is_most_probable_in_group
                })

        return pd.DataFrame(final_rows)