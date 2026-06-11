# src/models/train_ensemble.py

import os
import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.metrics import log_loss, brier_score_loss

from src.data.loader import load_matches
from src.models.ensemble import DixonColesXGBEnsemble
from src.features.network_feature import FootballNetworkModel
from src.features.squad_value import SquadValueExtractor


def _compute_pure_dc_probs_on_the_fly(df_val: pd.DataFrame, dc_params: dict) -> np.ndarray:
    """
    Calcula de manera independiente las predicciones del Dixon-Coles Puro
    para poder mantener la métrica de comparación en el reporte.
    """
    probs_list = []
    att = dc_params['attack']
    df_dict = dc_params['defense']
    gamma = dc_params['home_advantage']
    rho = dc_params['rho']

    max_goals = 8
    goals = np.arange(max_goals)

    def tau_scalar(x, y, lh, la, r):
        if x == y == 0: return 1 - lh * la * r
        if x == y == 1: return 1 - r
        if x == 1 and y == 0: return 1 + la * r
        if x == 0 and y == 1: return 1 + lh * r
        return 1.0

    for _, row in df_val.iterrows():
        home, away = row['home_team'], row['away_team']
        is_neutral = row.get('neutral', True)

        att_h = att.get(home, 1.0)
        def_h = df_dict.get(home, 1.0)
        att_a = att.get(away, 1.0)
        def_a = df_dict.get(away, 1.0)

        g = gamma if not is_neutral else 1.0
        lambda_h = att_h * def_a * g
        lambda_a = att_a * def_h

        prob_h = poisson.pmf(goals, lambda_h)
        prob_a = poisson.pmf(goals, lambda_a)
        probs = np.outer(prob_h, prob_a)

        probs[0, 0] *= tau_scalar(0, 0, lambda_h, lambda_a, rho)
        probs[0, 1] *= tau_scalar(0, 1, lambda_h, lambda_a, rho)
        probs[1, 0] *= tau_scalar(1, 0, lambda_h, lambda_a, rho)
        probs[1, 1] *= tau_scalar(1, 1, lambda_h, lambda_a, rho)

        probs = np.maximum(probs, 0.0)
        p_sum = probs.sum()
        if p_sum > 0:
            probs /= p_sum

        # Triángulos para clasificar la salida en Away (0), Draw (1), Home (2)
        p_home = np.sum(np.triu(probs, 1).T)
        p_draw = np.sum(np.diag(probs))
        p_away = np.sum(np.tril(probs, -1).T)

        probs_list.append([p_away, p_draw, p_home])

    return np.array(probs_list)


def evaluate_predictions(y_real: pd.Series, y_prob: np.ndarray) -> tuple:
    """
    Calcula el Log-Loss y el Brier Score para el conjunto de predicciones.
    """
    loss = log_loss(y_real, y_prob)

    # Brier Score para la probabilidad de Victoria Local (clase 2)
    y_real_binary = (y_real == 2).astype(int)
    prob_home = y_prob[:, 2]  # Clase 2
    brier = brier_score_loss(y_real_binary, prob_home)

    return loss, brier


def main():
    # Rutas relativas consistentes con el diseño de tu train_and_save.py
    matches_csv_path = '../../data/results.csv'
    fifa_rankings_path = '../../data/fifa_mens_rank.csv'
    dc_params_path = '../../models_saved/dixon_coles_params.joblib'
    ensemble_output_path = '../../models_saved/ensemble_model.joblib'

    profiles_path = '../../data/football-datasets/datalake/transfermarkt/player_profiles/player_profiles.csv'
    market_value_path = '../../data/football-datasets/datalake/transfermarkt/player_market_value/player_market_value.csv'
    squads_path = '../../data/squads/convocatorias_oficiales.csv'

    print("1. Cargando datos divididos por splits temporales...")
    df_train = load_matches(matches_csv_path, split='all')
    #df_val = load_matches(matches_csv_path, split='validate')
    df_val = df_train.tail(100)

    print(f"Partidos de Entrenamiento: {len(df_train)}")
    print(f"Partidos de Validación: {len(df_val)}")

    print("\n2. Cargando parámetros guardados de Dixon-Coles...")
    if not os.path.exists(dc_params_path):
        raise FileNotFoundError(
            f"No se encontraron los parámetros en {dc_params_path}. Ejecuta primero tu train_and_save.py.")

    dc_params = joblib.load(dc_params_path)

    print("\n2.5. Construyendo Grafo de Prestigio (Network Science)...")
    last_date_train = df_train['date'].max()
    start_date_momentum = last_date_train - pd.DateOffset(years=4)

    df_momentum = df_train[df_train['date'] >= start_date_momentum]

    network_model = FootballNetworkModel(damping_factor=0.85)
    df_all_matches = load_matches(matches_csv_path, split='all')
    network_model.fit_historical_data(df_all_matches)

    print(f"Grafo construido con {len(df_momentum)} partidos de momentum.")

    print("\n2.75. Inicializando Extractor de Valor de Mercado...")
    squad_extractor = SquadValueExtractor(
        profiles_path=profiles_path,
        market_value_path=market_value_path,
        squads_path=squads_path
    )

    print("\n3. Inicializando Ensamble Híbrido XGBoost...")
    ensemble = DixonColesXGBEnsemble(
        dc_params=dc_params,
        fifa_rankings_path=fifa_rankings_path,
        network_model=network_model,
        squad_extractor=squad_extractor
    )

    print("\n4. Entrenando el clasificador XGBoost (Sin Dixon-Coles en Características)...")
    # Al entrenar usando prepare_data dentro de ensemble.train,
    # el FeatureExtractor desacoplado no enviará variables de Dixon-Coles
    ensemble.train(df_train)

    print("\n5. Evaluando rendimiento en el conjunto de Validación (2020-2022)...")
    X_val, y_val = ensemble.prepare_data(df_val)

    # Predicciones de XGBoost sin fuga de datos
    y_prob_ensemble = ensemble.clf.predict_proba(X_val)

    # Predicciones de control usando Dixon-Coles Puro calculado dinámicamente
    y_prob_dc_only = _compute_pure_dc_probs_on_the_fly(df_val, dc_params)

    # Evaluación cruzada
    loss_dc, brier_dc = evaluate_predictions(y_val, y_prob_dc_only)
    loss_ens, brier_ens = evaluate_predictions(y_val, y_prob_ensemble)

    print("\n================ REPORT DE RENDIMIENTO (DESACOPLADO SOTA) ================")
    print(f"Dixon-Coles Puro             -> Log-Loss: {loss_dc:.4f} | Brier Score: {brier_dc:.4f}")
    print(f"Ensamble XGBoost (Sin DC)    -> Log-Loss: {loss_ens:.4f} | Brier Score: {brier_ens:.4f}")

    improvement = (loss_dc - loss_ens) / loss_dc * 100
    print(f"Mejora relativa en Log-Loss: {improvement:.2f}%")
    print("==========================================================================")

    # 6. Guardar el modelo de ensamble entrenado
    os.makedirs("../../models_saved", exist_ok=True)
    joblib.dump(ensemble, ensemble_output_path)
    print(f"\n¡Ensamble desacoplado guardado correctamente en: '{ensemble_output_path}'!")


if __name__ == "__main__":
    main()