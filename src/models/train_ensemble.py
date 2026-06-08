# src/models/train_ensemble.py

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss

from src.data.loader import load_matches
from src.models.ensemble import DixonColesXGBEnsemble
from src.features.network_feature import FootballNetworkModel
from src.features.squad_value import SquadValueExtractor

def evaluate_predictions(y_real: pd.Series, y_prob: np.ndarray) -> tuple:
    """
    Calcula el Log-Loss y el Brier Score para el conjunto de predicciones.
    """
    # 1. Log-loss de tres clases (0, 1, 2)
    loss = log_loss(y_real, y_prob)

    # 2. Brier Score para la probabilidad de Victoria Local (clase 2)
    y_real_binary = (y_real == 2).astype(int)
    prob_home = y_prob[:, 2]  # Probabilidad de la clase 2 (Local)
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
    # Usamos tu cargador oficial
    df_train = load_matches(matches_csv_path, split='train')
    df_val = load_matches(matches_csv_path, split='validate')

    print(f"Partidos de Entrenamiento: {len(df_train)}")
    print(f"Partidos de Validación: {len(df_val)}")

    print("\n2. Cargando parámetros guardados de Dixon-Coles...")
    if not os.path.exists(dc_params_path):
        raise FileNotFoundError(
            f"No se encontraron los parámetros en {dc_params_path}. Ejecuta primero tu train_and_save.py.")

    dc_params = joblib.load(dc_params_path)

    print("\n2.5. Construyendo Grafo de Prestigio (Network Science)...")
    # Filtramos los últimos 4 años del set de entrenamiento para el PageRank
    # Esto asegura que el "prestigio" sea el momentum actual antes del 2020
    last_date_train = df_train['date'].max()
    start_date_momentum = last_date_train - pd.DateOffset(years=4)

    df_momentum = df_train[df_train['date'] >= start_date_momentum]

    network_model = FootballNetworkModel(damping_factor=0.85)
    df_all_matches=load_matches(matches_csv_path, split='all')
    network_model.fit_historical_data(df_all_matches)

   # network_model.build_graph(df_momentum)
   # network_model.calculate_centrality()
    print(f"Grafo construido con {len(df_momentum)} partidos de momentum.")
    # -----------------------------------------------

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
    print("\n4. Entrenando el clasificador XGBoost...")
    ensemble.train(df_train)

    print("\n5. Evaluando rendimiento en el conjunto de Validación (2020-2022)...")
    X_val, y_val = ensemble.prepare_data(df_val)

    # Predicciones con el Ensamble de ML
    y_prob_ensemble = ensemble.clf.predict_proba(X_val)

    # Predicciones de control usando únicamente Dixon-Coles Puro
    y_prob_dc_only = X_val[['dc_prob_away', 'dc_prob_draw', 'dc_prob_home']].values

    # Evaluación
    loss_dc, brier_dc = evaluate_predictions(y_val, y_prob_dc_only)
    loss_ens, brier_ens = evaluate_predictions(y_val, y_prob_ensemble)

    print("\n================ REPORT DE RENDIMIENTO ================")
    print(f"Dixon-Coles Puro  -> Log-Loss: {loss_dc:.4f} | Brier Score: {brier_dc:.4f}")
    print(f"Ensamble XGBoost  -> Log-Loss: {loss_ens:.4f} | Brier Score: {brier_ens:.4f}")

    improvement = (loss_dc - loss_ens) / loss_dc * 100
    print(f"Mejora relativa en Log-Loss: {improvement:.2f}%")
    print("======================================================")

    # 6. Guardar el modelo de ensamble entrenado
    os.makedirs("../../models_saved", exist_ok=True)
    joblib.dump(ensemble, ensemble_output_path)
    print(f"\n¡Ensamble guardado correctamente en: '{ensemble_output_path}'!")


if __name__ == "__main__":
    main()