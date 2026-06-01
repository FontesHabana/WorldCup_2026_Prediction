import pandas as pd

from src.config.hyperparameters import TEAMS

def load_matches(filepath, split='train', date_train_end='2020-01-01',
                 date_val_end='2022-12-31'):
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

    df['home_team'] = df['home_team'].str.lower()
    df['away_team'] = df['away_team'].str.lower()

    date_train_end = pd.to_datetime(date_train_end)
    date_val_end = pd.to_datetime(date_val_end)

    if split == 'train':
        df = df[df['date'] < date_train_end]
    elif split == 'validate':
        df = df[(df['date'] >= date_train_end) & (df['date'] < date_val_end)]
    elif split == 'test':
        df = df[df['date'] >= date_val_end]
    else:
        raise ValueError(f"split debe ser 'train', 'validate' o 'test', recibí: {split}")

    return df.reset_index(drop=True)