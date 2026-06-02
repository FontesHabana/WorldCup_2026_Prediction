# src/simulation/run_contrast.py

import os
import joblib
import pandas as pd
from src.simulation.monte_carlo import MonteCarloEngine
from src.utils.persistence import load_model


def main():
    # Rutas de entrada y salida
    dc_params_path = "../../models_saved/dixon_coles_params.joblib"
    ensemble_path = "../../models_saved/ensemble_model.joblib"

    output_dc_csv = "../../simulation_results_dc.csv"
    output_xgb_csv = "../../simulation_results_xgb.csv"

    print("=== CARGANDO PARÁMETROS BASE ===")
    try:
        params = load_model(dc_params_path)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo Dixon-Coles en {dc_params_path}.")
        return

    iterations = 100

    # ----------------------------------------------------
    # RUN 1: Simulación con Dixon-Coles Puro
    # ----------------------------------------------------
    print(f"\n[RUN 1] Iniciando simulación con Dixon-Coles Puro ({iterations} iteraciones)...")
    # Forzamos que 'ensemble' sea None para desactivar XGBoost
    params['ensemble'] = None

    engine_dc = MonteCarloEngine(iterations=iterations)
    engine_dc.run(params)

    df_dc = engine_dc.get_probability_report()
    df_dc.to_csv(output_dc_csv)
    print(f"¡Resultados Dixon-Coles guardados en '{output_dc_csv}'!")

    # ----------------------------------------------------
    # RUN 2: Simulación con Ensamble XGBoost Calibrado
    # ----------------------------------------------------
    print(f"\n[RUN 2] Iniciando simulación con Ensamble XGBoost ({iterations} iteraciones)...")
    try:
        ensemble = joblib.load(ensemble_path)
        params['ensemble'] = ensemble
    except FileNotFoundError:
        print(f"Error: No se encontró el ensamble en {ensemble_path}. Abortando Run 2.")
        return

    engine_xgb = MonteCarloEngine(iterations=iterations)
    engine_xgb.run(params)

    df_xgb = engine_xgb.get_probability_report()
    df_xgb.to_csv(output_xgb_csv)
    print(f"¡Resultados Ensamble XGBoost guardados en '{output_xgb_csv}'!")

    # ----------------------------------------------------
    # ANÁLISIS DE CONTRASTE COMPARATIVO
    # ----------------------------------------------------
    print("\n================ ANALISIS DE CONTRASTE (Top 10 Favoritos) ================")
    # Renombrar columnas para la fusión de datos
    df_dc_clean = df_dc[['Champion']].rename(columns={'Champion': 'Prob_DC_Champ'})
    df_xgb_clean = df_xgb[['Champion']].rename(columns={'Champion': 'Prob_XGB_Champ'})

    # Unir ambos resultados en base al índice del equipo
    df_comparison = df_dc_clean.join(df_xgb_clean, how='outer').fillna(0.0)

    # Calcular la diferencia absoluta en puntos porcentuales
    df_comparison['Diferencia_Neta (%)'] = df_comparison['Prob_XGB_Champ'] - df_comparison['Prob_DC_Champ']

    # Ordenar por el favorito de XGBoost y mostrar los 10 mejores
    df_comparison = df_comparison.sort_values(by='Prob_XGB_Champ', ascending=False)
    print(df_comparison.head(10))
    print("=========================================================================")
    print("Nota: Un valor positivo en 'Diferencia_Neta (%)' indica que el modelo de ML")
    print("considera que el equipo tiene mejor probabilidad que la predicha por Poisson puro.")


if __name__ == "__main__":
    main()