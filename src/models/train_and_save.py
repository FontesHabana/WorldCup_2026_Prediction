# train_and_save.py
import os
from src.data.loader import load_matches
from src.models.dixon_coles import fit  # Ajusta la ruta a tu archivo dixon_coles
from src.utils.persistence import save_model


def main():
    print("Cargando partidos históricos...")
    df = load_matches('../../datareview/results.csv')

    print("Ajustando modelo Dixon-Coles (Vectorizado)...")
    result = fit(df)

    if result['success']:
        print("¡Ajuste exitoso!")
        # Crear el directorio si no existe
        os.makedirs("../../models_saved", exist_ok=True)

        # Guardar los parámetros reales
        save_model(result, '../../models_saved/dixon_coles_params.joblib')
        print("Parámetros del modelo guardados correctamente en 'dixon_coles_params.joblib'.")
    else:
        print("Error: El optimizador no pudo converger. No se guardó el modelo.")


if __name__ == "__main__":
    main()