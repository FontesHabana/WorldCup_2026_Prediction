import os
import json
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from jedi.inference.syntax_tree import infer_trailer
from tqdm import tqdm
from collections import Counter
from src.simulation.monte_carlo import MonteCarloEngine
from src.utils.persistence import load_model
import src.simulation.match_simulator as ms

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
    home="argentina"
    away="spain"

    winH=0;
    winA=0;
    draw=0;
    res=[]

    for i in tqdm(range(iterations)):
        result=ms.simulate_match(home_name=home, away_name=away, model_params=params, neutral=True)

        if result.home_goals>result.away_goals: winH+=1
        elif result.away_goals>result.home_goals: winA+=1
        else: draw+=1
        res.append((result.home_goals, result.away_goals))



    # 5. Mostrar resultados clave en la consola
    frecuencia=Counter(res)
    masFrecuente,conteo=frecuencia.most_common(1)[0]

    print("\n" + "=" * 50)
    print("🏆 REPORT DE RESULTADOS MUNDIAL 2026 🏆")
    print("=" * 50)

    print("Home = " + str((winH/iterations)*100))
    print("Away = " + str((winA/iterations)*100))
    print("Draw = " + str((draw/iterations)*100))
    print("GOles" + str(masFrecuente))

if __name__ == "__main__":
    main()