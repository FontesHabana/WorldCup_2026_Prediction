import pandas as pd
import numpy as np


def group_calculator(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la tabla de posiciones por grupo a partir de un DataFrame de partidos.

    Parámetros:
    df (pd.DataFrame): DataFrame con las columnas 'group', 'home_name', 'away_name',
                       'home_code', 'away_code', 'home_emoji', 'away_emoji',
                       'home_color', 'away_color', y 'result'.

    Retorna:
    pd.DataFrame: Tabla de posiciones detallada por grupo.
    """
    # Extraer todos los equipos únicos con sus metadatos por grupo
    home_teams = df[['group', 'home_name', 'home_code', 'home_emoji', 'home_color']].rename(
        columns={'home_name': 'team_name', 'home_code': 'team_code', 'home_emoji': 'team_emoji',
                 'home_color': 'team_color'}
    )
    away_teams = df[['group', 'away_name', 'away_code', 'away_emoji', 'away_color']].rename(
        columns={'away_name': 'team_name', 'away_code': 'team_code', 'away_emoji': 'team_emoji',
                 'away_color': 'team_color'}
    )

    # Combinar y eliminar duplicados para asegurar que cada equipo figure una vez por grupo
    teams = pd.concat([home_teams, away_teams]).drop_duplicates(subset=['group', 'team_name']).copy()

    # Inicializar columnas de estadísticas
    teams['Pld'] = 0
    teams['W'] = 0
    teams['D'] = 0
    teams['L'] = 0
    teams['GF'] = 0
    teams['GA'] = 0
    teams['GD'] = 0
    teams['Pts'] = 0

    # Establecer índice temporal para actualizar datos de forma eficiente
    teams = teams.set_index(['group', 'team_name'])

    # Procesar los resultados de los partidos
    for _, row in df.iterrows():
        result = row['result']

        # Omitir si el partido no se ha jugado o no tiene un resultado válido
        if pd.isna(result) or not isinstance(result, str) or '-' not in result:
            continue

        try:
            home_score, away_score = map(int, result.split('-'))
        except ValueError:
            continue  # En caso de que el formato del resultado no sea numérico

        group = row['group']
        home_team = row['home_name']
        away_team = row['away_name']

        # Actualizar partidos jugados y goles
        teams.loc[(group, home_team), 'Pld'] += 1
        teams.loc[(group, home_team), 'GF'] += home_score
        teams.loc[(group, home_team), 'GA'] += away_score

        teams.loc[(group, away_team), 'Pld'] += 1
        teams.loc[(group, away_team), 'GF'] += away_score
        teams.loc[(group, away_team), 'GA'] += home_score

        # Determinar resultado y actualizar puntos, victorias, empates y derrotas
        if home_score > away_score:
            teams.loc[(group, home_team), 'W'] += 1
            teams.loc[(group, home_team), 'Pts'] += 3
            teams.loc[(group, away_team), 'L'] += 1
        elif home_score < away_score:
            teams.loc[(group, away_team), 'W'] += 1
            teams.loc[(group, away_team), 'Pts'] += 3
            teams.loc[(group, home_team), 'L'] += 1
        else:
            teams.loc[(group, home_team), 'D'] += 1
            teams.loc[(group, home_team), 'Pts'] += 1
            teams.loc[(group, away_team), 'D'] += 1
            teams.loc[(group, away_team), 'Pts'] += 1

    # Calcular la diferencia de goles
    teams['GD'] = teams['GF'] - teams['GA']

    # Restaurar el índice para convertir 'group' y 'team_name' de nuevo en columnas
    standings = teams.reset_index()

    # Ordenar por: Grupo (asc), Puntos (desc), Diferencia de goles (desc), Goles a favor (desc)
    standings = standings.sort_values(
        by=['group', 'Pts', 'GD', 'GF', 'team_name'],
        ascending=[True, False, False, False, True]
    )

    # Asignar la posición final dentro de cada grupo
    standings['Pos'] = standings.groupby('group').cumcount() + 1

    # Reorganizar el orden de las columnas para la presentación
    column_order = [
        'group', 'Pos', 'team_name', 'team_code', 'team_emoji',
        'team_color', 'Pld', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts'
    ]

    return standings[column_order].reset_index(drop=True)