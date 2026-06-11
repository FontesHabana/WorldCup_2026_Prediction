import os
import json
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
        print("✅ Parámetros Dixon-Coles cargados con éxito.")
    except FileNotFoundError:
        print("❌ Error: No se encontró el archivo de Dixon-Coles en '../../models_saved/dixon_coles_params.joblib'.")
        return

    # 2. Cargar ensamble guardado de XGBoost
    ensemble_path = "../../models_saved/ensemble_model.joblib"
    try:
        ensemble = joblib.load(ensemble_path)
        params['ensemble'] = ensemble
        print("✅ Ensamble de Machine Learning (XGBoost) integrado.")
    except FileNotFoundError:
        params['ensemble'] = None
        print("⚠️ Advertencia: No se encontró 'ensemble_model.joblib'. Simulando con Dixon-Coles puro.")

    # 3. Inicializar y correr el motor de Monte Carlo con altas iteraciones
    iterations = 100000
    engine = MonteCarloEngine(iterations=iterations)

    print(f"\n🚀 Iniciando simulación de Monte Carlo ({iterations} mundiales simulados)...")
    engine.run(params)

    # 4. Obtener reportes probabilísticos finales
    df_prog_report = engine.get_probability_report()
    df_group_report = engine.get_group_standings_report()
    bracket_mas_probable = engine.extract_most_probable_bracket()
    df_matches_per_group = engine.get_most_probable_matches_per_group_report()
    # NUEVO: Obtener reporte de Fase de Grupos más probable
    reporte_grupos_mas_probable = engine.get_most_probable_group_stage_report()

    # 5. Mostrar resultados clave en la consola
    print("\n" + "=" * 50)
    print("🏆 REPORT DE RESULTADOS MUNDIAL 2026 🏆")
    print("=" * 50)
    print("\n🔥 TOP 10 FAVORITOS PARA GANAR EL MUNDIAL:")
    print(df_prog_report[["Champion", "Finalist", "Semi-Final", "Quarter-Final"]].head(10))

    # NUEVO: Imprime todos los partidos de grupos con marcadores comunes y tablas resultantes
    engine.print_most_probable_tournament_summary()

    print("\n📋 CRUCES CLAVE DEL BRACKET MÁS PROBABLE (CON MARCADOR ESTIMADO):")
    for m_id in ["M73", "M89", "M97", "M101", "M103", "M104"]:
        if m_id in bracket_mas_probable:
            match_data = bracket_mas_probable[m_id]
            print(f"  * {m_id}: {match_data['matchup']} -> "
                  f"Marcador estimado: {match_data['predicted_score']} | "
                  f"Ganador Predicho: {match_data['predicted_winner']} (Prob: {match_data['winner_probability_pct']}%)")

    # 6. Guardar todos los reportes a disco duro
    os.makedirs("../../simulation_results", exist_ok=True)

    # Reporte de avance de fases
    df_prog_report.to_csv("../../simulation_results/probabilidades_fases.csv", index=True)
    # Reporte de posiciones de grupo
    df_group_report.to_csv("../../simulation_results/probabilidades_grupos.csv", index=False)
    df_matches_per_group.to_csv("../../simulation_results/partidos_mas_probables_por_grupo.csv", index=False)
    print("💾 Reporte de partidos más probables por grupo guardado en 'partidos_mas_probables_por_grupo.csv'.")
    # Bracket más probable (Cruces dinámicos)
    with open("../../simulation_results/bracket_mas_probable.json", "w", encoding="utf-8") as f:
        json.dump(bracket_mas_probable, f, indent=4, ensure_ascii=False)

    # NUEVO: Exportar partidos de fase de grupos mas probables y tablas de posiciones asociadas
    if reporte_grupos_mas_probable:
        # Guardar lista de partidos
        with open("../../simulation_results/partidos_grupos_mas_probables.txt", "w", encoding="utf-8") as f:
            for g_name, data in reporte_grupos_mas_probable.items():
                f.write(f"\n⚽ {g_name} ⚽\n")
                for match in data["matches"]:
                    f.write(f"  - {match}\n")

        # Guardar tablas de posiciones individuales por grupo en CSV
        for g_name, data in reporte_grupos_mas_probable.items():
            safe_name = g_name.replace(" ", "_").lower()
            data["table"].to_csv(f"../../simulation_results/tabla_mas_probable_{safe_name}.csv", index=False)

    print("\n💾 Resultados físicos exportados exitosamente en la carpeta 'simulation_results/'.")

    # 7. Graficar Progresión de los Favoritos
    plt.figure(figsize=(12, 8))
    plot_data = df_prog_report.head(10)

    sns.set_theme(style="whitegrid")
    ax = plot_data[['Champion', 'Finalist', 'Semi-Final', 'Quarter-Final']].plot(
        kind='barh',
        stacked=True,
        colormap='viridis',
        figsize=(12, 7)
    )

    plt.title(f"Mundial 2026 - Probabilidades Acumuladas ({iterations} iteraciones con Ensamble ML)")
    plt.xlabel("Probabilidad (%)")
    plt.ylabel("Selección")
    plt.legend(title="Fase alcanzada")
    plt.gca().invert_yaxis()  # Favorito arriba
    plt.tight_layout()
    plt.savefig("../../simulation_results/grafico_probabilidades.png")
    plt.show()


if __name__ == "__main__":
    main()