import numpy as np
from scipy.special import gammaln
from src.simulation.models import MatchResult

# =====================================================================
# CONSTANTES DE OPTIMIZACIÓN Y CONFIGURACIÓN DE BAJO NIVEL
# =====================================================================
MAX_GOALS_LIMIT = 8
GOALS_ARR = np.arange(MAX_GOALS_LIMIT)

# Precalculamos factoriales y sus logaritmos para evitar overhead en el bucle de Monte Carlo
FACTORIALS = np.array([1, 1, 2, 6, 24, 120, 720, 5040], dtype=float)
LN_FACTORIALS = np.log(FACTORIALS)

# Diccionario global para cachear las predicciones del modelo de Machine Learning (XGBoost)
# Esto reduce la complejidad temporal de la inferencia de O(N) a O(1) tras el primer cálculo.
_XGB_CACHE = {}


def clear_simulator_cache() -> None:
    """Limpia el caché de predicciones de XGBoost (útil entre ejecuciones independientes)."""
    global _XGB_CACHE
    _XGB_CACHE.clear()


def tau(x: int, y: int,
        lambda_home: float, lambda_away: float,
        rho: float) -> float:
    """
    Factor de corrección Dixon-Coles para resultados de pocos goles (Legacy / Retrocompatibilidad).
    Nota: En simulate_match se utiliza una implementación vectorizada equivalente para mayor velocidad.
    """
    if x == y == 0:
        return 1 - lambda_home * lambda_away * rho
    if x == y == 1:
        return 1 - rho
    if x == 1 and y == 0:
        return 1 + lambda_away * rho
    if x == 0 and y == 1:
        return 1 + lambda_home * rho
    return 1.0


def calculate_pmf(lambda_val: float, max_goals: int, alpha: float) -> np.ndarray:
    """
    Calcula la Probability Mass Function (PMF) para una matriz de goles de forma ultra rápida.
    Bypassea las lentas validaciones de tipos e instanciación de scipy.stats.
    """
    goals = np.arange(max_goals)

    # Manejo de casos de intensidad nula o casi nula
    if lambda_val <= 1e-10:
        pmf = np.zeros(max_goals)
        pmf[0] = 1.0
        return pmf

    # Fase de Grupos o simulaciones estándar (Poisson puro)
    if alpha <= 0.0:
        if max_goals <= MAX_GOALS_LIMIT:
            fact = FACTORIALS[:max_goals]
        else:
            fact = np.cumprod(np.concatenate(([1.0], np.arange(1.0, max_goals))))

        # Fórmula de Poisson: (lambda^k * exp(-lambda)) / k!
        return (lambda_val ** goals) * np.exp(-lambda_val) / fact

    # Eliminatoria directa con sobredispersión (Binomial Negativa)
    n = 1.0 / alpha
    p = n / (n + lambda_val)

    if max_goals <= MAX_GOALS_LIMIT:
        ln_fact = LN_FACTORIALS[:max_goals]
    else:
        ln_fact = gammaln(goals + 1.0)

    # Fórmula matemática de la Binomial Negativa optimizada usando log-gamma (gammaln)
    # PMF = exp( gammaln(k + n) - gammaln(n) - ln(k!) + n * ln(p) + k * ln(1 - p) )
    # Evaluamos (1 - p) directamente como (lambda_val / (n + lambda_val)) para estabilidad numérica.
    ln_pmf = (gammaln(goals + n) - gammaln(n) - ln_fact +
              n * np.log(p) + goals * np.log(lambda_val / (n + lambda_val)))

    return np.exp(ln_pmf)


def simulate_match(home_name: str, away_name: str, model_params: dict, neutral: bool = True) -> MatchResult:
    """
    Simula un partido de fútbol generando marcadores basados en Poisson/Binomial Negativa
    calibrando los resultados globales con predicciones de XGBoost en tiempo récord.
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
    electric_factor = model_params.get('electric_factor', 0.0)

    # 2. Generar distribuciones de goles de manera ultra-rápida
    prob_h = calculate_pmf(lambda_h, max_goals, electric_factor)
    prob_a = calculate_pmf(lambda_a, max_goals, electric_factor)
    probs = np.outer(prob_h, prob_a)

    # 3. Ajuste Dixon-Coles Vectorizado inline (Evita 4 llamadas costosas a la función 'tau')
    probs[0, 0] *= (1.0 - lambda_h * lambda_a * rho)
    probs[0, 1] *= (1.0 + lambda_h * rho)
    probs[1, 0] *= (1.0 + lambda_a * rho)
    probs[1, 1] *= (1.0 - rho)

    probs = np.maximum(probs, 0.0)
    probs_sum = probs.sum()
    if probs_sum > 0:
        probs /= probs_sum

    # --- ACOPLAMIENTO DE MACHINE LEARNING CON CACHÉ (XGBoost Rescaling) ---
    if 'ensemble' in model_params and model_params['ensemble'] is not None:
        ensemble = model_params['ensemble']

        # Búsqueda en el caché de predicciones para evitar inferencias repetidas en Monte Carlo
        cache_key = (home_name, away_name)
        if cache_key in _XGB_CACHE:
            p_home_xg, p_draw_xg, p_away_xg = _XGB_CACHE[cache_key]
        else:
            p_home_xg, p_draw_xg, p_away_xg = ensemble.predict_match_probs(home_name, away_name)
            _XGB_CACHE[cache_key] = (p_home_xg, p_draw_xg, p_away_xg)

        # Calcular probabilidades de Dixon-Coles actuales
        p_home_dc = np.sum(np.triu(probs, 1).T)
        p_draw_dc = np.sum(np.diag(probs))
        p_away_dc = np.sum(np.tril(probs, -1).T)

        p_home_dc = max(p_home_dc, 1e-6)
        p_draw_dc = max(p_draw_dc, 1e-6)
        p_away_dc = max(p_away_dc, 1e-6)

        # Creación eficiente de máscaras booleanas
        home_mask = np.triu(np.ones_like(probs), 1).T > 0
        away_mask = np.tril(np.ones_like(probs), -1).T > 0
        draw_mask = np.eye(max_goals, dtype=bool)

        # Re-escalar las secciones de la matriz según las predicciones de XGBoost
        probs[home_mask] *= (p_home_xg / p_home_dc)
        probs[draw_mask] *= (p_draw_xg / p_draw_dc)
        probs[away_mask] *= (p_away_xg / p_away_dc)

        probs = np.maximum(probs, 0.0)
        probs_sum = probs.sum()
        if probs_sum > 0:
            probs /= probs_sum

    # 4. Muestreo de bajo nivel usando ravel() (vistas rápidas de memoria)
    res_idx = np.random.choice(max_goals * max_goals, p=probs.ravel())
    home_goals = res_idx // max_goals
    away_goals = res_idx % max_goals

    return MatchResult(
        home_team=home_name,
        away_team=away_name,
        home_goals=int(home_goals),
        away_goals=int(away_goals)
    )


def get_match_electric_factor(stage: str) -> float:
    """
    Retorna el factor de sobredispersión (alpha) según la instancia del Mundial 2026.
    A mayor alpha, mayor probabilidad de marcadores atípicos (goleadas o empates caóticos).
    """
    stage = stage.lower().strip()

    if stage == "group_stage":
        return 0.02  # Un toque mínimo de sobredispersión natural

    elif stage in ["round_of_32", "round_of_16"]:
        return 0.08

    elif stage in ["quarter_finals", "semi_finals"]:
        return 0.06

    elif stage in ["final", "third_place"]:
        return 0.03

    return 0.0