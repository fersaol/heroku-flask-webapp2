import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

print("Módulo General Listo Para Usarse \U0001F4BB")



def dataframes_charger(filename):
    """Función que importa el csv deseado desde el directorio
    
    -------------------------------------
    # Args:
       filename: (str)

    -------------------------------------
    # Return:
        pd.DataFrame"""

    current_path = Path.cwd()/"data/processed/"
    data = pd.read_csv(current_path/filename)
    return data

def models_saver(object,filename):

    """Función para guardar los modelos de machine learning elegidos en .pkl
    
    -----------------------------------
    # Args:
        object: objeto con el modelo de machine learning entrenado
        filename: (str) nombre del archivo pickle a guardar
        
    -----------------------------------
    # Return
        Guarda el modelo en formato pickle (.pkl)"""

    destino = Path(os.getcwd().replace("notebooks","model"))
    joblib.dump(value=object,filename=destino/f"{filename}.pkl")
    print("Modelo guardado correctamente")