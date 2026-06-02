# src/models/ensemble.py

import os
import scipy.stats as stats
import numpy as np
import pandas as pd
from typing import Tuple, Dict
from xgboost import XGBClassifier
from src.features.extractor import FeatureExtractor


def predict_probabilities_from_params(home_team: str, away_team: str, dc_params: dict) -> Tuple[float, float, float]:
    """
    Calcula las probabilidades de (Victoria Local, Empate, Victoria Visitante)
    usando la distribución de Poisson y los parámetros optimizados de Dixon-Coles.
    """
    try:
        # Extraemos los parámetros del diccionario de tu fit() de Dixon-Coles
        alpha_home = dc_params['attack'][home_team]
        beta_home = dc_params['defence'][home_team]

        alpha_away = dc_params['attack'][away_team]
        beta_away = dc_params['defence'][away_team]

        home_adv = dc_params['home_adv']
        rho = dc_params['rho']
    except KeyError:
        # Si un equipo no tiene parámetros registrados en tu Dixon-Coles,
        # asignamos una probabilidad neutra uniforme para evitar fallos.
        return 0.38, 0.26, 0.36

    # 1. Calcular las tasas de goles esperados (lambdas)
    lambda_home = np.exp(alpha_home + beta_away + home_adv)
    lambda_away = np.exp(alpha_away + beta_home)

    # 2. Generar la matriz de distribución conjunta de goles
    max_goals = 10
    prob_matrix = np.zeros((max_goals + 1, max_goals + 1))

    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            p_x = stats.poisson.pmf(x, lambda_home)
            p_y = stats.poisson.pmf(y, lambda_away)
            prob_matrix[x, y] = p_x * p_y

            # Ajuste de dependencia de bajos goles (Dixon-Coles rho)
            if rho != 0:
                if x == 0 and y == 0:
                    prob_matrix[x, y] *= (1 - lambda_home * lambda_away * rho)
                elif x == 1 and y == 0:
                    prob_matrix[x, y] *= (1 + lambda_home * rho)
                elif x == 0 and y == 1:
                    prob_matrix[x, y] *= (1 + lambda_away * rho)
                elif x == 1 and y == 1:
                    prob_matrix[x, y] *= (1 - rho)

    # 3. Sumar probabilidades correspondientes a cada resultado
    prob_home = float(np.sum(np.triu(prob_matrix, 1).T))  # Goles Local > Goles Visitante
    prob_away = float(np.sum(np.tril(prob_matrix, -1).T))  # Goles Visitante > Goles Local
    prob_draw = float(np.sum(np.diag(prob_matrix)))  # Goles Local == Goles Visitante

    total = prob_home + prob_draw + prob_away
    return prob_home / total, prob_draw / total, prob_away / total


class DixonColesXGBEnsemble:
    """
    Modelo de Ensamble Híbrido.
    Integra las probabilidades de tu Dixon-Coles con un clasificador XGBoost.
    """

    def __init__(self, dc_params: Dict, fifa_rankings_path: str):
        self.dc_params = dc_params
        self.feature_extractor = FeatureExtractor(fifa_rankings_path)

        # Clasificador XGBoost con hiperparámetros estables
        self.clf = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            objective='multi:softprob',
            eval_metric='mlogloss',
            random_state=42
        )

    def prepare_data(self, df_matches: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepara la matriz de diseño (features) y el vector objetivo (targets).
        """
        X_features = []
        y_targets = []

        for _, row in df_matches.iterrows():
            home = row['home_team']
            away = row['away_team']

            # Obtenemos las probabilidades de Dixon-Coles
            p_home, p_draw, p_away = predict_probabilities_from_params(home, away, self.dc_params)

            # Extraemos las características del partido
            features = self.feature_extractor.extract_features(
                home_team=home,
                away_team=away,
                dc_prob_home=p_home,
                dc_prob_draw=p_draw,
                dc_prob_away=p_away
            )

            # Mapeamos el resultado real: 2 = Local, 1 = Empate, 0 = Visitante
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
        """
        Entrena el clasificador XGBoost usando los partidos históricos.
        """
        X_train, y_train = self.prepare_data(df_train_matches)
        self.clf.fit(X_train, y_train)

    def predict_match_probs(self, home_team: str, away_team: str) -> Tuple[float, float, float]:
        """
        Genera la predicción final corregida del ensamble (Home, Draw, Away).
        """
        p_home, p_draw, p_away = predict_probabilities_from_params(home_team, away_team, self.dc_params)

        features = self.feature_extractor.extract_features(
            home_team=home_team,
            away_team=away_team,
            dc_prob_home=p_home,
            dc_prob_draw=p_draw,
            dc_prob_away=p_away
        )

        df_features = pd.DataFrame([features])
        probs = self.clf.predict_proba(df_features)[0]

        return float(probs[2]), float(probs[1]), float(probs[0])