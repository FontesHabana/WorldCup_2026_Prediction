import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

# Importaciones locales basadas en tu estructura
from src.config.hyperparameters import TEAMS
from src.simulation.match_simulator import calculate_pmf
from src.utils.persistence import load_model
from src.data.loader import load_matches


# =====================================================================
# 1. MÉTRICAS DE CALIDAD
# =====================================================================

def calculate_multiclass_brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula el Brier Score Multiclase (3 clases: Local, Empate, Visitante).
    Fórmula: (1 / N) * Sum_i( Sum_c( (p_ic - y_ic)^2 ) )
    """
    return float(np.mean(np.sum((y_pred - y_true) ** 2, axis=1)))


def calculate_multiclass_log_loss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    """
    Calcula el Log-Loss Multiclase recortando probabilidades extremas
    para estabilidad numérica.
    """
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
    y_pred_clipped /= np.sum(y_pred_clipped, axis=1, keepdims=True)
    loss = -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))
    return float(loss)


# =====================================================================
# 3. EXTRACCIÓN ANALÍTICA DE PROBABILIDADES
# =====================================================================

def predict_match_probabilities(
        home_name: str,
        away_name: str,
        model_params: dict,
        neutral: bool = True
) -> Tuple[float, float, float]:
    """
    Extrae analíticamente las probabilidades de victoria, empate y derrota
    reproduciendo fielmente la matemática de tu simulador.
    """
    att_h = model_params['attack'].get(home_name, 1.0)
    def_h = model_params['defense'].get(home_name, 1.0)
    att_a = model_params['attack'].get(away_name, 1.0)
    def_a = model_params['defense'].get(away_name, 1.0)
    gamma = model_params['home_advantage'] if not neutral else 1.0
    rho = model_params['rho']

    lambda_h = att_h * def_a * gamma
    lambda_a = att_a * def_h

    max_goals = 8
    # Usamos tensión estándar para amistosos de preparación del backtesting (0.04)
    electric_factor = model_params.get('electric_factor', 0.04)

    # Distribuciones de goles vía tu calculate_pmf optimizado
    prob_h = calculate_pmf(lambda_h, max_goals, electric_factor)
    prob_a = calculate_pmf(lambda_a, max_goals, electric_factor)
    probs = np.outer(prob_h, prob_a)

    # Ajuste Dixon-Coles Vectorizado inline
    probs[0, 0] *= (1.0 - lambda_h * lambda_a * rho)
    probs[0, 1] *= (1.0 + lambda_h * rho)
    probs[1, 0] *= (1.0 + lambda_a * rho)
    probs[1, 1] *= (1.0 - rho)

    probs = np.maximum(probs, 0.0)
    probs_sum = probs.sum()
    if probs_sum > 0:
        probs /= probs_sum

    # Acoplamiento del Ensemble de XGBoost si existe
    if 'ensemble' in model_params and model_params['ensemble'] is not None:
        ensemble = model_params['ensemble']
        p_home_xg, p_draw_xg, p_away_xg = ensemble.predict_match_probs(home_name, away_name)

        # Probabilidades base
        p_home_dc = max(np.sum(np.triu(probs, 1).T), 1e-6)
        p_draw_dc = max(np.sum(np.diag(probs)), 1e-6)
        p_away_dc = max(np.sum(np.tril(probs, -1).T), 1e-6)

        # Máscaras
        home_mask = np.triu(np.ones_like(probs), 1).T > 0
        away_mask = np.tril(np.ones_like(probs), -1).T > 0
        draw_mask = np.eye(max_goals, dtype=bool)

        # Re-escalamiento
        probs[home_mask] *= (p_home_xg / p_home_dc)
        probs[draw_mask] *= (p_draw_xg / p_draw_dc)
        probs[away_mask] *= (p_away_xg / p_away_dc)

        probs = np.maximum(probs, 0.0)
        probs_sum = probs.sum()
        if probs_sum > 0:
            probs /= probs_sum

    # Extracción de probabilidades agregadas
    p_draw = float(np.trace(probs))
    p_home = float(np.sum(np.tril(probs, k=-1)))  # Triangular inferior = Goles Local > Goles Visitante
    p_away = float(np.sum(np.triu(probs, k=1)))  # Triangular superior = Goles Visitante > Goles Local

    total = p_home + p_draw + p_away
    return p_home / total, p_draw / total, p_away / total


# =====================================================================
# 4. RUNNER DE BACKTESTING
# =====================================================================

def run_backtesting(filepath: str, model_params: dict) -> Dict[str, float]:
    """
    Ejecuta el backtesting sobre los partidos de prueba en el período de validación.
    """
    df_test = load_matches(
        filepath=filepath,
        split='validate',
        date_train_end='2025-01-01',
        date_val_end='2026-06-10'  # Coherente con la fecha de la sesión [1]
    )

    if df_test.empty:
        raise ValueError("❌ El set de pruebas está vacío. Revisa tus datos e intervalo de fechas.")

    print(f"📈 Procesando backtesting de {len(df_test)} partidos reales (2025-2026)...")

    y_true_list = []
    y_pred_list = []

    for _, row in df_test.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        home_score = row['home_score']
        away_score = row['away_score']

        # Vector One-hot real
        if home_score > away_score:
            y_true = [1.0, 0.0, 0.0]
        elif home_score == away_score:
            y_true = [0.0, 1.0, 0.0]
        else:
            y_true = [0.0, 0.0, 1.0]
        y_true_list.append(y_true)

        # Probabilidades analíticas del modelo híbrido
        p_home, p_draw, p_away = predict_match_probabilities(
            home_name=home_team,
            away_name=away_team,
            model_params=model_params,
            neutral=True  # Ajustar a False si evalúas ventaja de localía pura en eliminatorias
        )
        y_pred_list.append([p_home, p_draw, p_away])

    y_true_arr = np.array(y_true_list)
    y_pred_arr = np.array(y_pred_list)

    brier = calculate_multiclass_brier_score(y_true_arr, y_pred_arr)
    log_loss = calculate_multiclass_log_loss(y_true_arr, y_pred_arr)

    return {
        "brier_score": brier,
        "log_loss": log_loss,
        "evaluated_matches": len(df_test)
    }


# =====================================================================
# 5. PUNTO DE ENTRADA / ORQUESTACIÓN
# =====================================================================

if __name__ == "__main__":
    # Ajusta estas rutas a tu estructura local de directorios
    MODEL_PARAMS_PATH = "../../models_saved/dixon_coles_params.joblib"
    ENSEMBLE_PATH = "../../models_saved/ensemble_model.joblib"
    MATCHES_CSV_PATH = "../../data/results.csv"

    print("--- INICIANDO DIAGNÓSTICO DEL MOTOR ---")

    # 1. Cargar coeficientes de Dixon-Coles
    try:
        params = load_model(MODEL_PARAMS_PATH)
        print("✅ Parámetros Dixon-Coles cargados con éxito.")
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo Dixon-Coles en: {MODEL_PARAMS_PATH}")
        params = None

    if params:
        # 2. Cargar Ensemble (XGBoost)
        try:
            ensemble = joblib.load(ENSEMBLE_PATH)
            params['ensemble'] = ensemble
            print("✅ Ensamble XGBoost cargado e integrado.")
        except FileNotFoundError:
            params['ensemble'] = None
            print("⚠️ Advertencia: No se encontró el archivo del ensamble. Corriendo con Dixon-Coles puro.")

        # 3. Lanzar Backtesting
        if os.path.exists(MATCHES_CSV_PATH):
            try:
                results = run_backtesting(MATCHES_CSV_PATH, params)
                print("\n" + "=" * 50)
                print("🏆 CONTROL DE CALIDAD DEL MOTOR (BACKTESTING 2025-2026) 🏆")
                print("=" * 50)
                print(f"Partidos reales evaluados: {results['evaluated_matches']}")
                print(f"Brier Score global:        {results['brier_score']:.5f}")
                print(f"Log-Loss multiclase:       {results['log_loss']:.5f}")
                print("=" * 50)
                print("💡 Nota: Un Log-Loss cercano a 1.0 indica un poder predictivo competitivo en fútbol.")
            except Exception as e:
                print(f"❌ Error durante el cómputo de validación: {e}")
        else:
            print(f"❌ No se encontró la base de datos de partidos en: {MATCHES_CSV_PATH}")
            print("Por favor, asegúrate de colocar tu archivo de partidos y actualizar 'MATCHES_CSV_PATH'.")