import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.simulation.monte_carlo import MonteCarloEngine
from src.utils.persistence import load_model


def main():
    # 1. Load your trained model parameters
    # Adjust path if necessary
    try:
        params = load_model("../../models_saved/dixon_coles_params.joblib")
        print("--- PARÁMETROS EN EL ARCHIVO JOBLIB ---")
        print(f"¿Existe 'spain'?: {'spain' in params['attack']}")
        if 'spain' in params['attack']:
            print(f"Ataque de España guardado: {params['attack']['spain']:.4f}")
            print(f"Ataque de Arabia Saudita guardado: {params['attack']['saudi arabia']:.4f}")
    except FileNotFoundError:
        print("Model file not found. Ensure the path is correct.")
        return

    # 2. Initialize and Run Engine
    # 1000 iterations is a good balance between speed and statistical significance
    iterations = 1000
    engine = MonteCarloEngine(iterations=iterations)

    print(f"Starting Monte Carlo Simulation ({iterations} iterations)...")
    engine.run(params)

    # 3. Get the probability report
    df_report = engine.get_probability_report()

    # 4. Display Top 15 Favorites
    print("\n--- TOURNAMENT PROBABILITIES (Top 15) ---")
    print(df_report.head(15))

    # 5. Visualization
    plt.figure(figsize=(12, 8))
    # Taking top 10 for the plot
    plot_data = df_report.head(10)

    sns.set_theme(style="whitegrid")
    ax = plot_data[['Champion', 'Finalist', 'Semi-Final']].plot(
        kind='barh',
        stacked=True,
        colormap='viridis',
        figsize=(10, 6)
    )

    plt.title(f"Tournament Win Probability (Based on {iterations} simulations)")
    plt.xlabel("Probability (%)")
    plt.ylabel("Team")
    plt.legend(title="Stage Reached")
    plt.gca().invert_yaxis()  # Highest probability at the top
    plt.tight_layout()
    plt.show()

    # 6. Optional: Save to CSV
    # df_report.to_csv("simulation_results.csv")


if __name__ == "__main__":
    main()