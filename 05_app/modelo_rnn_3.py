# your_model.py
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Definir la ruta al archivo del modelo
model_path = '../04_models/01_primer_modelo.keras'
pkl_path = '../04_models//02_modelo_1_and_objects.pkl'

# Verificar si los archivos existen
if not os.path.exists(model_path):
    raise ValueError(f"El archivo no se encuentra en la ruta: {model_path}")
if not os.path.exists(pkl_path):
    raise ValueError(f"El archivo no se encuentra en la ruta: {pkl_path}")

# Cargar el modelo
model = load_model(model_path)

# Cargar el diccionario con el vectorizador y otros objetos
with open(pkl_path, 'rb') as f:
    objects_to_save = pickle.load(f)

# Obtener el vectorizador
vectorizer = objects_to_save['vectorizer']

# Definir la función de predicción
def predict_comment(comment):
    # Vectorizar el comentario
    input_text = vectorizer(([comment]))
    probabilities = model.predict(np.expand_dims(input_text, 0))  # Realizar la predicción
    # Asumimos que la salida es un array con las probabilidades para cada clase
    labels = ["Toxic", "Severe Toxic", "Obscene", "Threat", "Insult", "Identity Hate"]
    
    # Filtrar etiquetas con valor mayor a 0.5
    result = {label: prob for label, prob in zip(labels, probabilities[0]) if prob > 0.5}
    
    return result
