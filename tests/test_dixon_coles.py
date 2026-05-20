# tests/test_dixon_coles.py
import numpy as np
import pytest
from src.models.dixon_coles import (
    poisson_prob, tau, neg_log_likelihood,
    initialize_params, fit
)


# TEST 1 — poisson_prob con valor conocido
def test_poisson_prob_known_value():
    # P(X=0 | λ=1.5) = e^(-1.5) = 0.2231
    result = poisson_prob(1.5, 0)
    assert abs(result - 0.2231) < 0.001


# TEST 2 — tau en caso normal (x=2, y=2) siempre debe ser 1.0
def test_tau_no_correction_high_scores():
    result=tau(2,2,1,1,0.1)
    assert result==1.0


# TEST 3 — tau nunca debe retornar valor negativo con rho en bounds
def test_tau_never_negative():
    # Prueba con lambdas típicas y rho máximo permitido (0.2)
    result=tau(0,0,0.5,0.5,0.2)
    assert result>=0


# TEST 4 — initialize_params retorna vector de longitud correcta
def test_initialize_params_length():
    teams = ["Brazil", "Argentina", "France", "Germany"]
    # Con 4 equipos: 4 + 4 + 1 + 1 = 10 parámetros
    result=initialize_params(teams)
    assert len(result)==10


# TEST 5 — fit converge con datos sintéticos mínimos
def test_fit_runs_without_error():
    teams = ["TeamA", "TeamB"]
    matches = [
        {"home_team": "TeamA", "away_team": "TeamB",
         "home_goals": 2, "away_goals": 1, "weight": 1.0},
        {"home_team": "TeamB", "away_team": "TeamA",
         "home_goals": 0, "away_goals": 0, "weight": 0.8},
        {"home_team": "TeamA", "away_team": "TeamB",
         "home_goals": 1, "away_goals": 1, "weight": 0.9},
    ]
    result = fit(matches, teams)
    assert result['success']==True and len(result['params'])==6