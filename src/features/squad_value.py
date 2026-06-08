# src/features/squad_value.py

import pandas as pd
import numpy as np
from typing import Dict, Optional


class SquadValueExtractor:
    """
    Extrae el valor de mercado promedio por línea de un plantel.

    Fuentes de datos:
        - player_profiles.csv          → posición de cada jugador
        - player_market_value.csv      → historial de valuaciones con fecha
        - convocatorias_oficiales.csv  → plantel exacto por torneo

    Imputación:
        Jugadores sin valuación en Transfermarkt reciben el percentil 10
        del dataset global — conservador pero no cero.
    """

    POSITION_MAP = {
        # Porteros
        'Goalkeeper':                    'gk',
        # Defensas
        'Defender':                      'def',
        'Defender - Centre-Back':        'def',
        'Defender - Left-Back':          'def',
        'Defender - Right-Back':         'def',
        # Centrocampistas
        'Midfield':                      'mid',
        'Midfield - Central Midfield':   'mid',
        'Midfield - Attacking Midfield': 'mid',
        'Midfield - Defensive Midfield': 'mid',
        'Midfield - Left Midfield':      'mid',
        'Midfield - Right Midfield':     'mid',
        # Delanteros
        'Attack':                        'fwd',
        'Attack - Centre-Forward':       'fwd',
        'Attack - Left Winger':          'fwd',
        'Attack - Right Winger':         'fwd',
        'Attack - Second Striker':       'fwd',
    }

    # Mapeo posición del CSV de convocatorias → línea
    SQUAD_POSITION_MAP = {
        'GK':  'gk',
        'DEF': 'def',
        'MID': 'mid',
        'FWD': 'fwd',
    }

    def __init__(self,
                 profiles_path: str,
                 market_value_path: str,
                 squads_path: str):
        """
        Parámetros
        ----------
        profiles_path      : player_profiles.csv
        market_value_path  : player_market_value.csv
        squads_path        : convocatorias_oficiales.csv
                             Columnas: team_name, player_name, position,
                                       player_id, tournament, tournament_date
        """

        # ── 1. Perfiles → posición por línea ──────────────────────────────
        profiles = pd.read_csv(
            profiles_path,
            usecols=['player_id', 'main_position']
        )
        profiles['line'] = profiles['main_position'].map(self.POSITION_MAP)
        self.profiles = profiles.set_index('player_id')

        # ── 2. Historial de valores de mercado ────────────────────────────
        mv = pd.read_csv(
            market_value_path,
            usecols=['player_id', 'date_unix', 'value']
        )
        mv = mv.rename(columns={'date_unix': 'date'})
        mv['date'] = pd.to_datetime(mv['date'])
        mv = mv.dropna(subset=['value'])
        mv = mv[mv['value'] > 0]
        mv = mv.sort_values(['player_id', 'date'])  # CRÍTICO para merge_asof
        self.market_values = mv

        # ── 3. Percentil 10 global — imputación para desconocidos ─────────
        # Jugadores sin perfil en Transfermarkt no valen 0,
        # son simplemente jugadores de nivel bajo sin datos.
        self.p10_value = float(mv['value'].quantile(0.10))
        print(f"[SquadValueExtractor] p10_value = €{self.p10_value:,.0f}")

        # ── 4. Convocatorias oficiales ─────────────────────────────────────
        squads = pd.read_csv(squads_path)
        squads['tournament_date'] = pd.to_datetime(squads['tournament_date'])
        squads['team_name'] = squads['team_name'].str.strip().str.lower()

        # Añadir línea desde la columna 'position' del CSV de convocatorias
        squads['line'] = squads['position'].map(self.SQUAD_POSITION_MAP)

        self.squads = squads

    def _get_values_at_date(self,
                             player_ids: list,
                             match_date: pd.Timestamp) -> Dict[int, float]:
        """
        Para cada player_id, devuelve su valor de mercado más reciente
        ANTERIOR a match_date.

        Jugadores sin historial → imputa self.p10_value.

        Retorna dict {player_id: value}
        """
        if not player_ids:
            return {}

        # Filtramos solo los jugadores del plantel para eficiencia
        mv_squad = self.market_values[
            self.market_values['player_id'].isin(player_ids)
        ].copy()

        if mv_squad.empty:
            # Ningún jugador del plantel tiene historial → todos al p10
            return {pid: self.p10_value for pid in player_ids}

        # DataFrame con todos los player_ids y la fecha del partido
        query = pd.DataFrame({
            'player_id': player_ids,
            'date': match_date
        }).sort_values('date')

        # merge_asof: para cada jugador, último valor ANTES de match_date
        valued = pd.merge_asof(
            query,
            mv_squad.sort_values('date'),
            on='date',
            by='player_id',
            direction='backward'
        )

        result = {}
        for _, row in valued.iterrows():
            pid = int(row['player_id'])
            if pd.isna(row['value']):
                # Jugador existe en profiles pero sin valuación previa → p10
                result[pid] = self.p10_value
            else:
                result[pid] = float(row['value'])

        # Jugadores que no aparecieron en mv_squad en absoluto → p10
        for pid in player_ids:
            if pid not in result:
                result[pid] = self.p10_value

        return result

    def get_squad_value_by_line(self,
                                 team_name: str,
                                 match_date: str) -> Dict[str, float]:
        """
        Retorna el valor medio por línea del plantel en la fecha del partido.

        Parámetros
        ----------
        team_name  : nombre del equipo (ej. 'France', 'Brazil')
        match_date : 'YYYY-MM-DD'

        Retorna
        -------
        {
            'value_gk':         float,
            'value_def':        float,
            'value_mid':        float,
            'value_fwd':        float,
            'value_squad_mean': float,
            'n_players':        int    ← útil para debugging
        }
        """
        match_ts = pd.Timestamp(match_date)
        team_key = team_name.strip().lower()

        # ── A. Plantel más cercano anterior al partido ────────────────────
        team_squads = self.squads[
            (self.squads['team_name'] == team_key) &
            (self.squads['tournament_date'] <= match_ts)
        ]

        if team_squads.empty:
            return self._fallback_values()

        # Torneo más reciente antes del partido
        latest_tournament = (
            team_squads
            .sort_values('tournament_date')
            .iloc[-1]['tournament']
        )

        squad_df = self.squads[
            (self.squads['team_name'] == team_key) &
            (self.squads['tournament'] == latest_tournament)
        ][['player_id', 'line']].copy()

        if squad_df.empty:
            return self._fallback_values()

        player_ids = squad_df['player_id'].dropna().astype(int).tolist()

        # ── B. Valores históricos con imputación p10 ──────────────────────
        value_map = self._get_values_at_date(player_ids, match_ts)

        squad_df = squad_df.dropna(subset=['player_id'])
        squad_df['player_id'] = squad_df['player_id'].astype(int)
        squad_df['value'] = squad_df['player_id'].map(value_map)

        # Fallback de línea: si un jugador no tiene línea del CSV de convocatorias,
        # intentamos obtenerla de player_profiles
        missing_line = squad_df['line'].isna()
        if missing_line.any():
            for idx in squad_df[missing_line].index:
                pid = squad_df.loc[idx, 'player_id']
                if pid in self.profiles.index:
                    squad_df.loc[idx, 'line'] = self.profiles.loc[pid, 'line']
                else:
                    squad_df.loc[idx, 'line'] = 'mid'  # fallback final

        # ── C. Media por línea ────────────────────────────────────────────
        line_means = squad_df.groupby('line')['value'].mean()
        global_mean = float(squad_df['value'].mean())

        return {
            'value_gk':         float(line_means.get('gk',  self.p10_value)),
            'value_def':        float(line_means.get('def', self.p10_value)),
            'value_mid':        float(line_means.get('mid', self.p10_value)),
            'value_fwd':        float(line_means.get('fwd', self.p10_value)),
            'value_squad_mean': global_mean,
            'n_players':        len(squad_df),
        }

    def _fallback_values(self) -> Dict[str, float]:
        """
        Valores cuando no hay datos del equipo en absoluto.
        Usa p10 como estimación conservadora.
        """
        return {
            'value_gk':         self.p10_value,
            'value_def':        self.p10_value,
            'value_mid':        self.p10_value,
            'value_fwd':        self.p10_value,
            'value_squad_mean': self.p10_value,
            'n_players':        0,
        }

    def get_squad_values_batch(self,
                                team_name: str,
                                match_date: str) -> Dict[str, float]:
        """
        Alias público para consistencia con el FeatureExtractor.
        """
        return self.get_squad_value_by_line(team_name, match_date)