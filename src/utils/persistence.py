import joblib
import os

def save_model(model_params, filename="../../models_saved/dixon_coles_params.joblib"):
    """Guarda los parámetros del modelo en la carpeta models_saved."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    joblib.dump(model_params, filename)
    print(f"✅ Modelo guardado en {filename}")

def load_model(filename="models_saved/dixon_coles_params.joblib"):
    """Carga los parámetros del modelo."""
    if os.path.exists(filename):
        print(f"📂 Cargando modelo desde {filename}...")
        return joblib.load(filename)
    else:
        print("❌ No se encontró el archivo del modelo.")
        return None

