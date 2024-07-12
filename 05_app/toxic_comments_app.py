# toxic_comment_app.py

import streamlit as st
from modelo_rnn_3 import predict_comment

# Crear la interfaz de Streamlit
st.title("Clasificador de Comentarios Tóxicos")
st.write("Introduce un comentario para verificar sus niveles de toxicidad en diferentes etiquetas.")

# Campo de texto para el comentario
user_input = st.text_area("Comentario")

# Botón para realizar la predicción
if st.button("Clasificar Comentario"):
    if user_input:
        # Obtener predicciones del modelo
        predictions = predict_comment(user_input)
        st.write("Etiquetas asignadas con un valor mayor a 0.5:")
        for label, score in predictions.items():
            st.write(f"{label}: {score:.4f}")
    else:
        st.warning("Por favor, introduce un comentario.")