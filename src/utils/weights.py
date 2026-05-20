import numpy as np
import pandas as pd
from src.config.hyperparameters import (
    KAPPA,XI_INIT,BASE_DATE
)

def compute_weights(dates,competitions):

    dates=pd.to_datetime(dates)
    base_date = pd.to_datetime(BASE_DATE)
    d = np.array([(base_date - date).days for date in dates])
    kappa_vals = np.array([KAPPA.get(c, KAPPA["default"]) for c in competitions])
    w = kappa_vals * np.exp(-XI_INIT * d)
    return w
