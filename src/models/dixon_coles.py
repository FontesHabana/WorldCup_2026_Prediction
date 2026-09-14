import numpy as np
import pandas as pd
from scipy.optimize import minimize
from src.config.hyperparameters import (
    BOUNDS_ATTACK, BOUNDS_DEFENSE, BOUNDS_GAMMA, BOUNDS_RHO
)
from utils.weights import compute_weights
from data.loader import load_matches


def neg_log_likelihood_vectorized(params: np.ndarray,
                                  n_teams: int,
                                  home_indices: np.ndarray,
                                  away_indices: np.ndarray,
                                  home_goals: np.ndarray,
                                  away_goals: np.ndarray,
                                  weights: np.ndarray,
                                  m00: np.ndarray,
                                  m11: np.ndarray,
                                  m10: np.ndarray,
                                  m01: np.ndarray) -> float:
    """
    Versión vectorizada de la verosimilitud negativa de Dixon-Coles.
    Calcula todos los partidos de forma simultánea usando arrays de NumPy.
    """
    attack = params[:n_teams]
    defense = params[n_teams:2 * n_teams]
    gamma = params[2 * n_teams]
    rho = params[2 * n_teams + 1]

    # Fuerza esperada de goles usando indexación avanzada
    lambda_home = attack[home_indices] * defense[away_indices] * gamma
    lambda_away = attack[away_indices] * defense[home_indices]

    # Inicializar el factor de corrección tau con unos
    tau_val = np.ones_like(home_goals, dtype=float)

    # Aplicar la corrección Dixon-Coles usando las máscaras precalculadas
    tau_val[m00] = 1.0 - lambda_home[m00] * lambda_away[m00] * rho
    tau_val[m11] = 1.0 - rho
    tau_val[m10] = 1.0 + lambda_away[m10] * rho
    tau_val[m01] = 1.0 + lambda_home[m01] * rho

    # Evitar indeterminaciones matemáticas (log de cero o valores negativos)
    tau_val = np.clip(tau_val, 1e-10, None)
    lambda_home = np.clip(lambda_home, 1e-10, None)
    lambda_away = np.clip(lambda_away, 1e-10, None)

    # Densidad de Poisson simplificada (omitimos constantes que no afectan la optimización)
    # log(Poisson) = k * log(lambda) - lambda
    log_poisson_home = home_goals * np.log(lambda_home) - lambda_home
    log_poisson_away = away_goals * np.log(lambda_away) - lambda_away
    log_tau = np.log(tau_val)

    # Suma ponderada por los pesos temporales
    weighted_log_lik = weights * (log_tau + log_poisson_home + log_poisson_away)

    return -np.sum(weighted_log_lik)


def fit(df: pd.DataFrame) -> dict:
    """
    Ajusta Dixon-Coles a alta velocidad usando verosimilitud vectorizada.
    """
    # 1. Asegurar mapeo único de equipos
    teams = sorted(list(pd.unique(df[['home_team', 'away_team']].values.ravel())))
    n = len(teams)
    team_index = {team: i for i, team in enumerate(teams)}

    # 2. Calcular pesos temporales
    df = df.reset_index(drop=True)
    weights = compute_weights(df['date'], df['tournament'])

    # 3. Convertir columnas de pandas a arrays de NumPy de alta velocidad
    home_indices = df['home_team'].map(team_index).values
    away_indices = df['away_team'].map(team_index).values
    home_goals = df['home_score'].values
    away_goals = df['away_score'].values

    # 4. Precalcular máscaras booleanas para la corrección tau (Dixon-Coles)
    m00 = (home_goals == 0) & (away_goals == 0)
    m11 = (home_goals == 1) & (away_goals == 1)
    m10 = (home_goals == 1) & (away_goals == 0)
    m01 = (home_goals == 0) & (away_goals == 1)

    # Limitar límites para evitar que el optimizador busque valores absurdos
    bounds = (
            [BOUNDS_ATTACK] * n +
            [BOUNDS_DEFENSE] * n +
            [BOUNDS_GAMMA] +
            [BOUNDS_RHO]
    )

    # Punto de partida uniforme
    params_init = np.concatenate([
        np.ones(n),  # Ataque inicial
        np.ones(n),  # Defensa inicial
        np.array([1.3]),  # Gamma inicial
        np.array([0.1])  # Rho inicial
    ])





    # Llamada al optimizador (ahora correrá órdenes de magnitud más rápido)
    result = minimize(
        fun=neg_log_likelihood_vectorized,
        x0=params_init,
        args=(n, home_indices, away_indices, home_goals, away_goals, weights, m00, m11, m10, m01),
        method="L-BFGS-B",
        bounds=bounds
    )



    attack = dict(zip(teams, result.x[:n]))
    defense = dict(zip(teams, result.x[n:2 * n]))
    gamma = float(result.x[2 * n])
    rho = float(result.x[2 * n + 1])

    return {
        'attack': attack,
        'defense': defense,
        'home_advantage': gamma,
        'rho': rho,
        'success': bool(result.success),
    }