import os
import scipy.stats as stats
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from xgboost import XGBClassifier
from src.features.extractor import FeatureExtractor


def predict_probabilities_from_params(home_team: str, away_team: str, dc_params: dict) -> Tuple[float, float, float]:
    """
    Calculates the baseline (Home Win, Draw, Away Win) probabilities
    using Poisson distributions and optimized Dixon-Coles parameters.
    """
    try:
        alpha_home = dc_params['attack'][home_team]
        beta_home = dc_params['defence'][home_team]

        alpha_away = dc_params['attack'][away_team]
        beta_away = dc_params['defence'][away_team]

        home_adv = dc_params['home_adv']
        rho = dc_params['rho']
    except KeyError:
        # Fallback values representing historical international match averages
        # if teams do not exist yet in the Dixon-Coles parameter dictionary.
        return 0.38, 0.26, 0.36

    lambda_home = np.exp(alpha_home + beta_away + home_adv)
    lambda_away = np.exp(alpha_away + beta_home)

    max_goals = 10
    prob_matrix = np.zeros((max_goals + 1, max_goals + 1))

    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            p_x = stats.poisson.pmf(x, lambda_home)
            p_y = stats.poisson.pmf(y, lambda_away)
            prob_matrix[x, y] = p_x * p_y

            # Apply low-scoring draw adjustment (Dixon-Coles rho coupling parameter)
            if rho != 0:
                if x == 0 and y == 0:
                    prob_matrix[x, y] *= (1 - lambda_home * lambda_away * rho)
                elif x == 1 and y == 0:
                    prob_matrix[x, y] *= (1 + lambda_home * rho)
                elif x == 0 and y == 1:
                    prob_matrix[x, y] *= (1 + lambda_away * rho)
                elif x == 1 and y == 1:
                    prob_matrix[x, y] *= (1 - rho)

    prob_home = float(np.sum(np.triu(prob_matrix, 1).T))
    prob_away = float(np.sum(np.tril(prob_matrix, -1).T))
    prob_draw = float(np.sum(np.diag(prob_matrix)))

    total = prob_home + prob_draw + prob_away
    if total == 0:
        return 0.38, 0.26, 0.36

    return prob_home / total, prob_draw / total, prob_away / total


class DixonColesXGBEnsemble:
    """
    SOTA Hybrid Ensemble Model.
    Integra Dixon-Coles, Rankings FIFA y Prestigio de Red (PageRank) dinámico
    en un clasificador XGBoost con alineación de características garantizada.
    """

    def __init__(self, dc_params: Dict, fifa_rankings_path: str, network_model=None, squad_extractor=None):
        self.dc_params = dc_params

        # Pasamos el modelo de red y squad extractor para generar las métricas dinámicas
        self.feature_extractor = FeatureExtractor(
            fifa_rankings_path,
            network_model=network_model,
            squad_extractor=squad_extractor
        )

        self.clf = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            objective='multi:softprob',
            eval_metric='mlogloss',
            random_state=42
        )

        self.feature_names: List[str] = []

    def prepare_data(self, df_matches: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepara la matriz de diseño (features) y el vector objetivo (targets).
        """
        X_features = []
        y_targets = []

        for _, row in df_matches.iterrows():
            home = row['home_team']
            away = row['away_team']

            # Manejo del tiempo/fechas
            if 'date' in row:
                match_date = str(row['date'])
                match_year = pd.to_datetime(row['date']).year
            else:
                match_date = "2026-06-11"
                match_year = 2026

            # 1. Probabilidades base de Dixon-Coles
            p_home, p_draw, p_away = predict_probabilities_from_params(home, away, self.dc_params)

            # 2. Extracción de características avanzadas (Network & Squad values)
            features = self.feature_extractor.extract_features(
                home_team=home,
                away_team=away,
                match_year=match_year,
                match_date=match_date,
                dc_prob_home=p_home,
                dc_prob_draw=p_draw,
                dc_prob_away=p_away
            )

            # Target Mapping: 2: Home Win, 1: Draw, 0: Away Win
            if row['home_score'] > row['away_score']:
                target = 2
            elif row['home_score'] == row['away_score']:
                target = 1
            else:
                target = 0

            X_features.append(features)
            y_targets.append(target)

        return pd.DataFrame(X_features), pd.Series(y_targets)

    def train(self, df_train_matches: pd.DataFrame):
        """Entrena el clasificador XGBoost y guarda el orden de las columnas."""
        X_train, y_train = self.prepare_data(df_train_matches)

        # Guardamos el orden exacto de las columnas de entrenamiento
        self.feature_names = X_train.columns.tolist()

        self.clf.fit(X_train, y_train)
        print(f"Ensemble entrenado exitosamente con {len(self.feature_names)} variables.")

    def predict_match_probs(self, home_team: str, away_team: str, match_year: int = 2026,
                            match_date: str = "2026-06-11") -> Tuple[float, float, float]:
        """
        Genera la predicción final usando el ensamble completo, previniendo
        errores de orden de columnas.
        """
        p_home, p_draw, p_away = predict_probabilities_from_params(home_team, away_team, self.dc_params)

        features = self.feature_extractor.extract_features(
            home_team=home_team,
            away_team=away_team,
            match_year=match_year,
            match_date=match_date,
            dc_prob_home=p_home,
            dc_prob_draw=p_draw,
            dc_prob_away=p_away
        )

        df_features = pd.DataFrame([features])

        # Forzar alineación y orden exacto de columnas
        if self.feature_names:
            df_features = df_features.reindex(columns=self.feature_names, fill_value=0.0)

        probs = self.clf.predict_proba(df_features)[0]

        # Retorna probabilidades correspondientes a: (Home, Draw, Away)
        return float(probs[2]), float(probs[1]), float(probs[0])