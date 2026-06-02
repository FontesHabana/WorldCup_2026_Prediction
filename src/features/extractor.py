# src/features/extractor.py

import pandas as pd
import numpy as np
from typing import Dict, Any


class FeatureExtractor:
    """
    Clase para construir variables de entrada (features) adaptadas al dataset FIFA.
    Columnas del CSV: ['date', 'semester', 'rank', 'team', 'acronym',
                       'total.points', 'previous.points', 'diff.points']
    """

    def __init__(self, fifa_rankings_path: str):
        # 1. Cargamos el dataset
        df_rankings = pd.read_csv(fifa_rankings_path)

        # 2. Estandarizamos los nombres de las columnas a minúsculas para evitar errores sintácticos
        df_rankings.columns = [c.strip().lower() for c in df_rankings.columns]

        # 3. Estandarizamos los nombres de los equipos
        df_rankings['team'] = df_rankings['team'].str.lower().str.strip()

        # --- EXPLICACIÓN METODOLÓGICA (Redundancia Temporal) ---
        # Como este dataset tiene un historial de rankings, el mismo equipo aparece varias veces.
        # Para nuestras predicciones del Mundial, nos interesa el estado físico más reciente de cada equipo.
        # Ordenamos por fecha (descendente) y eliminamos los registros duplicados de cada selección.
        df_rankings = df_rankings.sort_values(by='date', ascending=False)
        df_rankings = df_rankings.drop_duplicates(subset=['team'])

        # 4. Establecemos 'team' como el índice para búsquedas en tiempo constante O(1)
        self.rankings = df_rankings.set_index('team')

    def get_team_metrics(self, team: str) -> Dict[str, Any]:
        """
        Recupera el ranking, puntos totales y momento de forma de un equipo.
        Retorna valores neutros de penalización si el equipo no está registrado.
        """
        team_clean = team.lower().strip()
        try:
            row = self.rankings.loc[team_clean]
            return {
                'rank': int(row['rank']),
                'points': float(row['total.points']),
                'diff_points': float(row['diff.points'])
            }
        except KeyError:
            # Pedagogía: Si una selección pequeña clasificada al Mundial no está en el dataset,
            # evitamos retornar nulos (NaN) usando valores por defecto realistas.
            return {
                'rank': 120,  # Fuera del top 100
                'points': 900.0,  # Puntuación FIFA baja por defecto
                'diff_points': 0.0  # Sin tendencia de crecimiento conocida
            }

    def extract_features(self,
                         home_team: str,
                         away_team: str,
                         dc_prob_home: float,
                         dc_prob_draw: float,
                         dc_prob_away: float) -> Dict[str, Any]:
        """
        Construye la matriz de características para un enfrentamiento específico.
        """
        home_data = self.get_team_metrics(home_team)
        away_data = self.get_team_metrics(away_team)

        # --- INGENIERÍA DE VARIABLES (Explicación del "por qué") ---

        # 1. Diferencia de Rango Ordinal
        # Útil para árboles de decisión simples.
        rank_diff = away_data['rank'] - home_data['rank']

        # 2. Diferencia de Puntos FIFA (Métrica Continua)
        # Es superior al 'rank' porque los puestos en la tabla no son lineales.
        # La distancia en nivel futbolístico real entre el puesto 1 y el 2 es distinta
        # a la distancia entre el 80 y el 81. Los puntos netos sí reflejan esto.
        points_diff = home_data['points'] - away_data['points']

        # 3. Diferencia de Tendencia (Momento de Forma)
        # Si un equipo viene sumando puntos y el otro viene perdiendo, el modelo detectará la inercia.
        trend_diff = home_data['diff_points'] - away_data['diff_points']

        # 4. Interacción Probabilidad-Puntos
        # Cruzamos la confianza de Dixon-Coles con la diferencia de puntos reales para que el modelo
        # aprenda dinámicamente cuándo desconfiar del modelo de Poisson.
        dc_points_interaction = dc_prob_home * points_diff

        return {
            'dc_prob_home': dc_prob_home,
            'dc_prob_draw': dc_prob_draw,
            'dc_prob_away': dc_prob_away,
            'rank_diff': rank_diff,
            'points_diff': points_diff,
            'trend_diff': trend_diff,
            'dc_points_interaction': dc_points_interaction
        }