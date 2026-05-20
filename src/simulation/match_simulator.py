import numpy as np
from scipy.stats import poisson
from src.simulation.models import MatchResult
from src.models.dixon_coles import tau





def simulate_match(home_name: str, away_name: str, model_params: dict, neutral: bool = True) -> MatchResult:
    # 1. Extract params
    att_h = model_params['attack'].get(home_name, 1.0)
    def_h = model_params['defense'].get(home_name, 1.0)
    att_a = model_params['attack'].get(away_name, 1.0)
    def_a = model_params['defense'].get(away_name, 1.0)
    gamma = model_params['home_advantage'] if not neutral else 1.0
    rho = model_params['rho']

    # 2.  Calculate Lambdas
    lambda_h = att_h * def_a * gamma
    lambda_a = att_a * def_h

    # 3. Build Probability Matrix
    max_goals = 8  # 10 is enough for World Cup
    probs = np.zeros((max_goals, max_goals))

    for i in range(max_goals):  # Home goals
        for j in range(max_goals):  # Away goals
            # Poisson probability for i goals
            prob_i = poisson.pmf(i, lambda_h)
            # Poisson probability for j goals
            prob_j = poisson.pmf(j, lambda_a)

            #  Combine them and multiply by the tau correction!
            p = prob_i * prob_j * tau(i, j, lambda_h, lambda_a, rho)
            probs[i, j] = max(0, p)  # Ensure no negative numbers

    probs /= probs.sum()  # Normalize

    # 4. Random selection
    res_idx = np.random.choice(range(max_goals ** 2), p=probs.flatten())
    home_goals = res_idx // max_goals
    away_goals = res_idx % max_goals

    return MatchResult(home_team=home_name, away_team=away_name, home_goals=int(home_goals),away_goals= int(away_goals))