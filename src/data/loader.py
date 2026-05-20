import pandas as pd

from src.config.hyperparameters import TEAMS

def load_matches(filepath):
    # returns: DataFrame con columnas estandarizadas
    df=pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])

    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=['home_score', 'away_score'])
    df['home_score'] = df['home_score'].astype(int)
    df['away_score'] = df['away_score'].astype(int)
    df=df[df['date']>='1994-01-01']

    mask_home = df['home_team'].str.lower().isin(TEAMS)
    mask_away = df['away_team'].str.lower().isin(TEAMS)


    df = df[mask_home & mask_away].copy()

    return df