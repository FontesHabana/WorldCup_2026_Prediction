import numpy as np
from scipy.stats import poisson
from src.simulation.models import MatchResult
#from src.models.dixon_coles import tau


def tau(x: int, y: int,
        lambda_home: float, lambda_away: float,
        rho: float) -> float:
    """Dixon-Coles correction factor for low-score outcomes."""
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

    # Calculamos todos los PMF de un solo golpe usando NumPy
    # Usamos scipy.stats.poisson.pmf pasándole un array completo
    prob_h = poisson.pmf(goals, lambda_h)
    prob_a = poisson.pmf(goals, lambda_a)

    # Crear la matriz de probabilidades mediante producto externo
    probs = np.outer(prob_h, prob_a)

    # Aplicar la corrección Dixon-Coles (tau) solo a los marcadores 0-0, 1-0, 0-1, 1-1
    # Esto es mucho más rápido que llamar a una función en cada celda
    probs[0, 0] *= tau(0, 0, lambda_h, lambda_a, rho)
    probs[0, 1] *= tau(0, 1, lambda_h, lambda_a, rho)
    probs[1, 0] *= tau(1, 0, lambda_h, lambda_a, rho)
    probs[1, 1] *= tau(1, 1, lambda_h, lambda_a, rho)

    # Normalizar (asegurar que sumen 1 y no haya negativos)
    probs = np.maximum(probs, 0)
    probs /= probs.sum()

    # Selección aleatoria rápida
    res_idx = np.random.choice(max_goals ** 2, p=probs.flatten())
    home_goals = res_idx // max_goals
    away_goals = res_idx % max_goals

    return MatchResult(home_team=home_name, away_team=away_name, home_goals=int(home_goals), away_goals=int(away_goals))