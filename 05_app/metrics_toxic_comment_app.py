import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl 
import matplotlib.pyplot as plt

# Importamos la función de predicción desde el script del modelo

from modelo_lr import predict_comment

# Creamos la interfaz de Streamlit

st.title("Clasificador de Comentarios Tóxicos")
st.write("Introduce un comentario para verificar sus niveles de toxicidad en diferentes etiquetas.")

# Campo de texto para el comentario

user_input = st.text_area("Comentario")

# Botón para realizar la predicción

if st.button("Clasificar Comentario"):
    if user_input:

        # Obtenemos predicciones del modelo

        predictions = predict_comment(user_input)
        st.write(predictions)  # Mostrar las predicciones para depuración
        labels = list(predictions.keys())
        scores = list(predictions.values())
        
        # Crear la gráfica de barras horizontales

        fig, ax = plt.subplots()
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, scores, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()  # Las etiquetas van de arriba hacia abajo
        ax.set_xlabel('Nivel de toxicidad')
        ax.set_title('Resultados de Clasificación')

        # Mostrar la gráfica en Streamlit

        st.pyplot(fig)
    else:
        st.warning("Por favor, introduce un comentario.")