# Cargar el modelo desde el archivo pickle
import os
import pickle
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

# Ruta al archivo pickle
pkl_path = '../04_models/04_tercer_modelo.pkl'

# Verificar si el archivo existe
if not os.path.exists(pkl_path):
    raise ValueError(f"El archivo no se encuentra en la ruta: {pkl_path}")

# Cargar el diccionario con el modelo y otros objetos
with open(pkl_path, 'rb') as f:
    objects_to_save = pickle.load(f)

# Obtener el vectorizador y el modelo
vectorizer = objects_to_save['vectorizer']
classifier = objects_to_save['classifier']

# Reconstruir el pipeline
pipeline = Pipeline([
    ('vectorizer', vectorizer),
    ('classifier', classifier)
])

# Definir la función de predicción
def predict_comment(comment):
    # Asegúrate de pasar una lista de documentos al pipeline
    probabilities = pipeline.predict_proba([comment])
    # Asumimos que la salida es una lista de arrays con las probabilidades para cada clase
    labels = ["Toxic", "Severe Toxic", "Obscene", "Threat", "Insult", "Identity Hate"]
    
    # Filtrar etiquetas con valor mayor a 0.5
    result = {label: prob[1] for label, prob in zip(labels, probabilities) if prob[1] > 0.5}
    
    return result

# Prueba el pipeline con un comentario
comentario = 'You are a piece of shit, I am going to kill you'
predicciones = predict_comment(comentario)
print(predicciones)
