import pandas as pd
from collections import Counter, defaultdict
from tqdm import tqdm
import numpy as np

from src.simulation.knockout import run_tournament_knockout
from src.simulation.group_stage import simulate_group_stage
from src.config.hyperparameters import GROUPS
# NUEVO: Importamos el agregador de estadísticas modularizado
from src.simulation.stats_aggregator import TournamentAggregator


class MonteCarloEngine:
    def __init__(self, iterations: int = 1000):
        self.iterations = iterations
        # NUEVO: Instanciamos el agregador para centralizar toda la lógica de reportes
        self.aggregator = TournamentAggregator()
        self.total_goals_series = []
        self.tournament_summaries = []

    def run(self, model_params):
        # Limpiar estadísticas acumuladas reinstanciando el agregador modularizado
        self.aggregator = TournamentAggregator()
        self.total_goals_series.clear()
        self.tournament_summaries.clear()

        for i in tqdm(range(self.iterations), desc="Running Tournament"):
            # 1. Ejecución de la Fase de Grupos
            group_results = simulate_group_stage(GROUPS, model_params)

            # 2. Ejecución del Cuadro Eliminatorio (Knockout)
            winner, ko_result, match_history = run_tournament_knockout(group_results, model_params)

            # 3. NUEVO: Registrar los resultados de esta iteración en el agregador modularizado
            self.aggregator.aggregate_iteration(group_results, ko_result, match_history)

            # 4. Guardar Estadísticas de Metadatos del Torneo
            self._record_metadata(i, winner, match_history)

    def _record_metadata(self, sim_id, champion, match_history):
        total_goals = sum(m['home_goals'] + m['away_goals'] for m in match_history)
        self.tournament_summaries.append({
            "sim_id": sim_id,
            "champion": champion,
            "total_goals": total_goals,
            "avg_goals_per_match": total_goals / len(match_history) if match_history else 0
        })

    def get_probability_report(self) -> pd.DataFrame:
        """Genera el reporte de avance de cada equipo en porcentaje, manteniendo compatibilidad con run_simulation.py."""
        df_agg = self.aggregator.generate_progression_report()

        # Mapear las llaves del agregador al formato esperado por el graficador y reportes existentes de la simulación
        rename_dict = {
            "RoundOf32": "Round-of-32",
            "RoundOf16": "Round-of-16",
            "QuarterFinalists": "Quarter-Final",
            "SemiFinalists": "Semi-Final",
            "Finalist": "Finalist",
            "Winner": "Champion"
        }
        df_agg = df_agg.rename(columns=rename_dict)

        # Asegurar el orden cronológico de las fases para graficado y visualización
        stages_order = ["Round-of-32", "Round-of-16", "Quarter-Final", "Semi-Final", "Finalist", "Champion"]

        for col in stages_order:
            if col not in df_agg.columns:
                df_agg[col] = 0.0

        # Establecer la columna 'Team' como el índice del DataFrame para mantener compatibilidad con el graficador
        if "Team" in df_agg.columns:
            df_agg = df_agg.set_index("Team")

        df_agg=df_agg[stages_order]
        return df_agg.round(2)


    def get_group_standings_report(self) -> pd.DataFrame:
        """Genera el reporte probabilístico de clasificación de grupos delegando al agregador."""
        return self.aggregator.generate_group_standings_report()

    def extract_most_probable_bracket(self) -> dict:
        """Determina el cruce exacto, marcador y el ganador más probable delegando al agregador."""
        return self.aggregator.extract_most_probable_bracket()

    def get_most_probable_group_stage_report(self) -> dict:
        """Retorna el reporte de fase de grupos más probable desde el agregador."""
        return self.aggregator.generate_most_probable_group_stage_report()

    def print_most_probable_tournament_summary(self):
        """Imprime el resumen en consola del escenario más probable del torneo."""
        self.aggregator.print_most_probable_tournament_summary()

    def get_most_probable_matches_per_group_report(self) -> pd.DataFrame:
        """Retorna el reporte de partidos más probables por grupo listo para exportar a CSV."""
        return self.aggregator.generate_most_probable_matches_per_group_report()