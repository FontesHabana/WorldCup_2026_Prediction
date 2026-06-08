# src/simulation/run_simulation.py

import os
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.simulation.monte_carlo import MonteCarloEngine
from src.utils.persistence import load_model

def main():
    # 1. Cargar parámetros optimizados de Dixon-Coles
    try:
        params = load_model("../../models_saved/dixon_coles_params.joblib")
        print("--- PARÁMETROS EN EL ARCHIVO JOBLIB (DIXON-COLES) ---")
        print(f"¿Existe 'spain'?: {'spain' in params['attack']}")
        if 'spain' in params['attack']:
            print(f"Ataque de España guardado: {params['attack']['spain']:.4f}")
            print(f"Ataque de Arabia Saudita guardado: {params['attack']['saudi arabia']:.4f}")
    except FileNotFoundError:
        print("Error: No se encontró el archivo de Dixon-Coles. Verifica la ruta.")
        return

    # --- NUEVO: ACOPLAMIENTO DE MACHINE LEARNING ---
    # Cargamos el ensamble guardado de XGBoost y lo inyectamos en el diccionario de parámetros
    ensemble_path = "../../models_saved/ensemble_model.joblib"
    try:
        ensemble = joblib.load(ensemble_path)
        params['ensemble'] = ensemble
        print("¡Ensamble de Machine Learning (XGBoost) cargado e integrado con éxito!")
    except FileNotFoundError:
        params['ensemble'] = None
        print("Advertencia: No se encontró 'ensemble_model.joblib'. Corriendo simulación con Dixon-Coles puro.")
    # -----------------------------------------------

    # 2. Inicializar y correr el motor de Monte Carlo
    # 1000 iteraciones es el balance ideal para este sprint
    iterations = 100
    engine = MonteCarloEngine(iterations=iterations)

    print(f"\nIniciando simulación de Monte Carlo con Ensamble Inteligente ({iterations} iteraciones)...")
    engine.run(params)

    # 3. Obtener reporte probabilístico final
    df_report = engine.get_probability_report()

    # 4. Mostrar el Top 15 de favoritos para ganar el Mundial 2026
    print("\n--- PROBABILIDADES DEL TORNEO (Top 15 con XGBoost) ---")
    print(df_report.head(15))

    # 5. Visualización
    plt.figure(figsize=(12, 8))
    # Tomamos el Top 10 para graficar
    plot_data = df_report.head(10)

    sns.set_theme(style="whitegrid")
    ax = plot_data[['Champion', 'Finalist', 'Semi-Final']].plot(
        kind='barh',
        stacked=True,
        colormap='viridis',
        figsize=(10, 6)
    )

    plt.title(f"Probabilidades del Torneo (Basado en {iterations} simulaciones con Ensamble ML)")
    plt.xlabel("Probabilidad (%)")
    plt.ylabel("Equipo")
    plt.legend(title="Fase alcanzada")
    plt.gca().invert_yaxis()  # El favorito se muestra arriba
    plt.tight_layout()
    plt.show()

    # 6. Opcional: Guardar reporte final a un archivo CSV
    df_report.to_csv("../../simulation_results_xgb.csv")
    print("\nResultados de simulación guardados con éxito en 'simulation_results_xgb.csv'.")


if __name__ == "__main__":
    main()