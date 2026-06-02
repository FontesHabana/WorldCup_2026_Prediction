# src/simulation/match_simulator.py

import numpy as np
from scipy.stats import poisson
from src.simulation.models import MatchResult


def tau(x: int, y: int,
        lambda_home: float, lambda_away: float,
        rho: float) -> float:
    """Factor de corrección Dixon-Coles para resultados de pocos goles."""
    if x == y == 0:
        return 1 - lambda_home * lambda_away * rho
    if x == y == 1:
        return 1 - rho
    if x == 1 and y == 0:
        return 1 + lambda_away * rho
    if x == 0 and y == 1:
        return 1 + lambda_home * rho
    return 1.0


def simulate_match(home_name: str, away_name: str, model_params: dict, neutral: bool = True) -> MatchResult:
    """
    Simula un partido de fútbol generando marcadores basados en Poisson
    pero calibrando los resultados globales con predicciones de XGBoost (si están disponibles).
    """
    # 1. Recuperar parámetros para Dixon-Coles
    att_h = model_params['attack'].get(home_name, 1.0)
    def_h = model_params['defense'].get(home_name, 1.0)
    att_a = model_params['attack'].get(away_name, 1.0)
    def_a = model_params['defense'].get(away_name, 1.0)
    gamma = model_params['home_advantage'] if not neutral else 1.0
    rho = model_params['rho']

    lambda_h = att_h * def_a * gamma
    lambda_a = att_a * def_h

    max_goals = 8
    goals = np.arange(max_goals)

    # 2. Construir matriz de distribución de Poisson base
    prob_h = poisson.pmf(goals, lambda_h)
    prob_a = poisson.pmf(goals, lambda_a)
    probs = np.outer(prob_h, prob_a)

    probs[0, 0] *= tau(0, 0, lambda_h, lambda_a, rho)
    probs[0, 1] *= tau(0, 1, lambda_h, lambda_a, rho)
    probs[1, 0] *= tau(1, 0, lambda_h, lambda_a, rho)
    probs[1, 1] *= tau(1, 1, lambda_h, lambda_a, rho)

    probs = np.maximum(probs, 0)
    probs /= probs.sum()

    # --- ACOPLAMIENTO DE MACHINE LEARNING (XGBoost Rescaling) ---
    # Si pasamos el ensamble de ML dentro de los parámetros, corregimos la matriz
    if 'ensemble' in model_params and model_params['ensemble'] is not None:
        ensemble = model_params['ensemble']

        # A. Predicción inteligente de XGBoost
        p_home_xg, p_draw_xg, p_away_xg = ensemble.predict_match_probs(home_name, away_name)

        # B. Calcular las probabilidades de victoria implícitas en la matriz de Poisson actual
        # Triángulo superior (Local > Visitante)
        p_home_dc = np.sum(np.triu(probs, 1).T)
        # Diagonal (Local == Visitante)
        p_draw_dc = np.sum(np.diag(probs))
        # Triángulo inferior (Local < Visitante)
        p_away_dc = np.sum(np.tril(probs, -1).T)

        # Evitamos divisiones por cero con un float mínimo (épsilon)
        p_home_dc = max(p_home_dc, 1e-6)
        p_draw_dc = max(p_draw_dc, 1e-6)
        p_away_dc = max(p_away_dc, 1e-6)

        # C. Crear máscaras lógicas para cada resultado en la matriz
        home_mask = np.triu(np.ones_like(probs), 1).T > 0
        away_mask = np.tril(np.ones_like(probs), -1).T > 0
        draw_mask = np.eye(probs.shape[0], dtype=bool)

        # D. Escalar cada sección de la matriz según la corrección del XGBoost
        probs[home_mask] *= (p_home_xg / p_home_dc)
        probs[draw_mask] *= (p_draw_xg / p_draw_dc)
        probs[away_mask] *= (p_away_xg / p_away_dc)

        # E. Re-normalizar la matriz corregida
        probs = np.maximum(probs, 0)
        probs /= probs.sum()

    # 3. Selección aleatoria del marcador final usando la matriz calibrada
    res_idx = np.random.choice(max_goals ** 2, p=probs.flatten())
    home_goals = res_idx // max_goals
    away_goals = res_idx % max_goals

    return MatchResult(
        home_team=home_name,
        away_team=away_name,
        home_goals=int(home_goals),
        away_goals=int(away_goals)
    )