import numpy as np
import pandas as pd

from src.utils.persistence import save_model, load_model
from scipy.optimize import minimize
from scipy.stats import poisson
from src.config.hyperparameters import (
    BOUNDS_ATTACK, BOUNDS_DEFENSE, BOUNDS_GAMMA, BOUNDS_RHO
)
from utils.weights import compute_weights
from data.loader import load_matches


def poisson_prob(lambda_: float, k: int) -> float:
    """Return the probability of scoring exactly k goals with rate lambda_."""
    return poisson.pmf(k, lambda_)



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


def neg_log_likelihood(params: np.ndarray,
                       teams: list,
                       matches: list) -> float:
    """
    Objective function for scipy.optimize.minimize.

    params contains, in order:
        - attack[i]  for each team i  (len = n_teams)
        - defense[i] for each team i  (len = n_teams)
        - gamma      (home advantage) (index = 2*n_teams)
        - rho        (DC correction)  (index = 2*n_teams + 1)

    Returns:
        Negative weighted log-likelihood as float.
    """
    n = len(teams)
    team_index = {team: i for i, team in enumerate(teams)}

    attack = params[:n]
    defense = params[n:2 * n]
    gamma = params[2 * n]
    rho = params[2 * n + 1]

    log_likelihood = 0.0

    for match in matches:
        h = team_index[match["home_team"]]
        a = team_index[match["away_team"]]
        x = match["home_goals"]
        y = match["away_goals"]
        w = match["weight"]


        lambda_home = attack[h] * defense[a] * gamma
        lambda_away = attack[a] * defense[h]
        t = tau(x, y, lambda_home, lambda_away, rho)


        log_p = (
            np.log(np.clip(t,1e-10,None))
            + np.log(np.clip(poisson_prob(lambda_home, x),1e-10,None))
            + np.log(np.clip(poisson_prob(lambda_away, y),1e-10,None))
        )

        log_likelihood += w * log_p

    return -log_likelihood


def initialize_params(teams: list) -> np.ndarray:
    """
    Genera el vector inicial de parámetros para el optimizador.

    Orden: [attack × n_teams | defense × n_teams | gamma | rho]
    """
    n = len(teams)

    attack_init = np.ones(n)  # Ataque inicial: 1.0 para todos
    defense_init = np.ones(n)  # Defensa inicial: 1.0 para todos
    gamma_init = np.array([1.3])  # Ventaja de local inicial
    rho_init = np.array([0.1])

    return np.concatenate([attack_init,defense_init,gamma_init,rho_init])




def fit(df: pd.DataFrame) -> dict:
    """
    Estima parámetros Dixon-Coles via MLE sobre datos históricos reales.

    Recibe:
        df: DataFrame limpio de load_matches() con columnas:
            date, home_team, away_team, home_score, away_score,
            tournament, neutral

    Retorna:
        {
            'attack':         dict equipo -> float,
            'defense':        dict equipo -> float,
            'home_advantage': float  (gamma),
            'rho':            float  (corrección Dixon-Coles),
            'success':        bool
        }
    """
    teams = list(pd.unique(df[['home_team', 'away_team']].values.ravel()))
    n = len(teams)

    df = df.reset_index(drop=True)
    w = compute_weights(df['date'], df['tournament'])


    matches = df.assign(weight=w).rename(columns={
        'home_score': 'home_goals',
        'away_score': 'away_goals'
    })[['home_team', 'away_team', 'home_goals', 'away_goals', 'weight']].to_dict('records')

    bounds = (
            [BOUNDS_ATTACK] * n  +  # lista de 32 tuplas iguales
            [BOUNDS_DEFENSE] * n +  # lista de 32 tuplas iguales
            [BOUNDS_GAMMA]       +  # lista de 1 tupla
            [BOUNDS_RHO]            # lista de 1 tupla
    )

    # Punto de inicio
    params_init = initialize_params(teams)

    # Llamada a scipy
    result = minimize(fun=neg_log_likelihood,
                      x0=params_init,
                      args=(teams, matches),
                      method="L-BFGS-B",
                      bounds=bounds
    )

    attack = dict(zip(teams, result.x[:n]))
    defense = dict(zip(teams, result.x[n:2 * n]))
    gamma = result.x[2 * n]
    rho = result.x[2 * n + 1]

    return {
        'attack': attack,
        'defense': defense,
        'home_advantage': gamma,
        'rho': rho,
        'success': result.success,
    }


df=load_matches('../../datareview/results.csv')



result = fit(df)
teams = list(result['attack'].keys())
print([result['attack'][t] for t in teams[:5]])   # primeros 5 ataques
print(f"rho: {result['rho']:.4f}")
print(f"success: {result['success']}")
save_model(result)