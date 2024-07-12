
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# Cargamos el diccionario con los elementos asociados al modelo

with open('../04_models/02_modelo_1_and_objects.pkl', 'rb') as f:
    objects_to_save = pickle.load(f)

# Cargamos el modelo y el vectorizador

model = tf.keras.models.load_model('../04_models/01_primer_modelo.keras')
vectorizer = objects_to_save['vectorizer']

# Definir la función de predicción

def predict_comment(comment):

    # Vectorizamos el comentario de entrada

    input_text = vectorizer([comment])
    probabilities = model.predict(np.expand_dims(input_text, 0)) 

    # Asumimos que la salida es un array con las probabilidades para cada clase
    
    labels = ["Toxic", "Severe Toxic", "Obscene", "Threat", "Insult", "Identity Hate"]
    
    # Convertir las probabilidades a un diccionario
    return dict(zip(labels, probabilities[0]))  # Usa [0] para acceder al primer (y único) conjunto de predicciones