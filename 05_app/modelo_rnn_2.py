import pickle
import tensorflow as tf
import numpy as np
import pandas as pd


# Cargar el diccionario del archivo .pkl y el modelo desde el archivo .keras
with open('../04_models/02_modelo_1_and_objects.pkl', 'rb') as f:
    saved_objects = pickle.load(f)

model_path = saved_objects['model_path']
vectorizer = saved_objects['vectorizer']
precision_value = saved_objects['precision']
recall_value = saved_objects['recall']

model = tf.keras.models.load_model('../04_models/01_primer_modelo.keras')

# Definir la función de predicción

def predict_comment(comment):

    # Vectorizamos el comentario de entrada

    input_text = vectorizer([comment])
    probabilities = model.predict(np.expand_dims(input_text, 0)) 

    # Asumimos que la salida es un array con las probabilidades para cada clase
    
    labels = ["Toxic", "Severe Toxic", "Obscene", "Threat", "Insult", "Identity Hate"]
    
    # Convertir las probabilidades a un diccionario
    return dict(zip(labels, probabilities[0]))  # Usa [0] para acceder al primer (y único) conjunto de predicciones

