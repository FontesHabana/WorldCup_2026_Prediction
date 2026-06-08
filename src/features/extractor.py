import pandas as pd
import numpy as np
from typing import Dict, Any


class FeatureExtractor:
    """
    Clase para construir variables de entrada (features) adaptadas al dataset FIFA,
    permitiendo la consulta de métricas según el año correspondiente al partido,
    además de integrar prestigio dinámico (PageRank) y valor de plantilla por líneas.
    """

    def __init__(self, fifa_rankings_path: str, network_model=None, squad_extractor=None):
        self.network_model = network_model
        self.squad_extractor = squad_extractor

        # 1. Cargamos el dataset de rankings FIFA
        df_rankings = pd.read_csv(fifa_rankings_path)

        # 2. Estandarizamos los nombres de las columnas
        df_rankings.columns = [c.strip().lower() for c in df_rankings.columns]

        # 3. Estandarizamos los nombres de los equipos
        df_rankings['team'] = df_rankings['team'].str.lower().str.strip()

        # 4. Convertimos a tipo fecha y extraemos el año
        df_rankings['date'] = pd.to_datetime(df_rankings['date'], errors='coerce')
        df_rankings['year'] = df_rankings['date'].dt.year

        # --- RESOLUCIÓN DE REDUNDANCIA TEMPORAL POR AÑO ---
        # Ordenamos descendentemente por fecha para conservar el registro más reciente de ese año.
        df_rankings = df_rankings.sort_values(by='date', ascending=False)
        df_rankings_cleaned = df_rankings.drop_duplicates(subset=['team', 'year'])

        # 5. Construimos una estructura de diccionario anidado {team: {year: metrics}}
        self.rankings_history = {}
        for _, row in df_rankings_cleaned.iterrows():
            team = row['team']
            year = int(row['year']) if not pd.isna(row['year']) else None

            if not team or year is None:
                continue

            if team not in self.rankings_history:
                self.rankings_history[team] = {}

            self.rankings_history[team][year] = {
                'rank': int(row['rank']),
                'points': float(row['total.points']),
                'diff_points': float(row['diff.points'])
            }

    def _get_default_metrics(self) -> Dict[str, Any]:
        """Retorna valores por defecto en caso de no encontrar registros."""
        return {
            'rank': 120,
            'points': 900.0,
            'diff_points': 0.0
        }

    def get_team_metrics(self, team: str, year: int) -> Dict[str, Any]:
        """
        Recupera el ranking, puntos totales y momento de forma de un equipo en un año específico.
        """
        team_clean = team.lower().strip()

        if team_clean not in self.rankings_history:
            return self._get_default_metrics()

        team_years = self.rankings_history[team_clean]

        # Intentamos obtener el año exacto solicitado
        if year in team_years:
            return team_years[year]

        # Estrategia de Fallback: buscar el año anterior más cercano disponible
        past_years = [y for y in team_years.keys() if y < year]
        if past_years:
            closest_year = max(past_years)
            return team_years[closest_year]

        # Si no hay registros previos al año solicitado, usamos el año más antiguo disponible
        closest_year = min(team_years.keys())
        return team_years[closest_year]

    def extract_features(self,
                         home_team: str,
                         away_team: str,
                         match_year: int,
                         match_date: str,
                         dc_prob_home: float,
                         dc_prob_draw: float,
                         dc_prob_away: float) -> Dict[str, Any]:
        """
        Construye la matriz de características para un enfrentamiento específico
        utilizando los datos correspondientes al año del partido, prestigio dinámico y plantilla.
        """
        network_diff = 0.0

        # FIXED: Ahora pasamos match_date para hacer cálculo de prestigio dinámico libre de fugas
        if self.network_model is not None:
            h_net = self.network_model.get_team_centrality(home_team, match_date)
            a_net = self.network_model.get_team_centrality(away_team, match_date)

            if h_net > 0 and a_net > 0:
                network_diff = h_net - a_net

        home_data = self.get_team_metrics(home_team, match_year)
        away_data = self.get_team_metrics(away_team, match_year)

        # 1. Diferencia de Rango Ordinal
        rank_diff = away_data['rank'] - home_data['rank']

        # 2. Diferencia de Puntos FIFA
        points_diff = home_data['points'] - away_data['points']

        # 3. Diferencia de Tendencia
        trend_diff = home_data['diff_points'] - away_data['diff_points']

        # 4. Interacción Probabilidad-Puntos
        dc_points_interaction = dc_prob_home * points_diff

        # 5. Inicialización de valores de plantilla
        dif_value_gk = 0.0
        dif_value_def = 0.0
        dif_value_mid = 0.0
        dif_value_fwd = 0.0
        dif_value_squad_mean = 0.0

        if self.squad_extractor is not None:
            h_vals = self.squad_extractor.get_squad_value_by_line(home_team, match_date)
            a_vals = self.squad_extractor.get_squad_value_by_line(away_team, match_date)

            dif_value_gk = h_vals['value_gk'] - a_vals['value_gk']
            dif_value_def = h_vals['value_def'] - a_vals['value_def']
            dif_value_mid = h_vals['value_mid'] - a_vals['value_mid']
            dif_value_fwd = h_vals['value_fwd'] - a_vals['value_fwd']
            dif_value_squad_mean = h_vals['value_squad_mean'] - a_vals['value_squad_mean']

        # Retornamos el diccionario completo alineado para XGBoost
        return {
            'dc_prob_home': dc_prob_home,
            'dc_prob_draw': dc_prob_draw,
            'dc_prob_away': dc_prob_away,
            'rank_diff': rank_diff,
            'points_diff': points_diff,
            'trend_diff': trend_diff,
            'dc_points_interaction': dc_points_interaction,

            # FIXED: Retornamos las variables de red calculadas
            'network_diff': network_diff,
            'dc_network_interaction': dc_prob_home * network_diff,

            # Variables de plantilla por líneas
            'dif_value_gk': dif_value_gk,
            'dif_value_def': dif_value_def,
            'dif_value_mid': dif_value_mid,
            'dif_value_fwd': dif_value_fwd,
            'dif_value_squad_mean': dif_value_squad_mean,
        }