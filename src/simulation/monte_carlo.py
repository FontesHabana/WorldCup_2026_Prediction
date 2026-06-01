import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from src.simulation.knockout import run_tournament_knockout
from src.simulation.group_stage import simulate_group_stage
from collections import Counter, defaultdict
from src.config.hyperparameters import GROUPS
import numpy as np

from src.utils.persistence import load_model



class MonteCarloEngine:
    def __init__(self, iterations: int = 1000):
        self.iterations = iterations
        self.team_stats = defaultdict(Counter)
        self.total_goals_series = []
        self.tournament_summaries = []

    def run(self, model_params):
        for i in tqdm(range(self.iterations), desc="Running Tournament"):
            # 1. Ejecución
            group_results = simulate_group_stage(GROUPS,model_params)
            winner,ko_result, match_history = run_tournament_knockout(group_results, model_params)

            # 2. Guardar Estadísticas de Equipos (Fácil de leer)
            self._update_team_stats(ko_result)

            # 3. Guardar Estadísticas del Torneo (Para análisis de goles)
            self._record_metadata(i, ko_result['Winner'], match_history)

    def _update_team_stats(self, ko_result):
        # 1. The Winner
        winner = ko_result['Winner']
        self.team_stats[winner]['Champion'] += 1

        # 2. Both Finalists (Winner + Runner up)
        for team in ko_result['Finalist']:
            self.team_stats[team]['Finalist'] += 1

        # 3. All SemiFinalists (Top 4)
        for team in ko_result['SemiFinalists']:
            self.team_stats[team]['Semi-Final'] += 1

        # 4. All QuarterFinalists (Top 8)
        for team in ko_result['QuarterFinalists']:
            self.team_stats[team]['Quarter-Final'] += 1

    def _record_metadata(self, sim_id, champion, match_history):
        # Esta es tu "Caja Negra" para entender si el simulador funciona bien
        total_goals = sum(m['home_goals'] + m['away_goals'] for m in match_history)
        self.tournament_summaries.append({
            "sim_id": sim_id,
            "champion": champion,
            "total_goals": total_goals,
            "avg_goals_per_match": total_goals / len(match_history)
        })

    def get_probability_report(self) -> pd.DataFrame:
        # 1. Convertimos el diccionario a un DataFrame de Pandas
        df = pd.DataFrame.from_dict(self.team_stats, orient='index').fillna(0)

        # 2. Calculamos porcentajes (Probabilidades)
        # Dividimos cada conteo por el número de iteraciones
        df_prob = (df / self.iterations) * 100

        # 3. Limpieza: Ordenar por quién gana más veces el mundial
        if 'Champion' in df_prob.columns:
            df_prob = df_prob.sort_values(by='Champion', ascending=False)

        return df_prob.round(2)  # Retornar con 2 decimales para elegancia





