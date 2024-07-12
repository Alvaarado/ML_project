import streamlit as st
import pickle
import tensorflow as tf
import numpy as np

# Cargar el modelo y el vectorizador

with open('../04_models/02_modelo_1_and_objects.pkl', 'rb') as f:
    saved_objects = pickle.load(f)

model = tf.keras.models.load_model('../04_models/01_primer_modelo.keras')
model_path = saved_objects['model_path']
vectorizer = saved_objects['vectorizer']
precision_value = saved_objects['precision']
recall_value = saved_objects['recall']

# Función para predecir la toxicidad

def predict_toxicity(comment):
    # Transformar el comentario usando el vectorizador
    vectorized_comment = vectorizer([comment])
    # Realizar la predicción
    prediction = model.predict(np.expand_dims(vectorized_comment,0))
    
    return prediction

# Configuración de la aplicación Streamlit
st.title("Clasificador de Toxicidad de Comentarios")
st.write("Introduce un comentario para verificar si encaja en alguna categoría de toxicidad.")

# Entrada del usuario
user_comment = st.text_area("Comentario:")

if st.button("Clasificar"):
    if user_comment:
        # Realizar la predicción
        prediction = predict_toxicity(user_comment)
        toxicity_level = prediction[0]  # Ajusta esto según cómo se estructure tu salida

        # Mostrar resultados
        st.write("Nivel de Toxicidad:")
        st.write(toxicity_level)  # Ajusta el formato según tu salida
    else:
        st.warning("Por favor, introduce un comentario.")
