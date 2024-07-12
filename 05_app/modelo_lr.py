import joblib

# Cargamos el pipeline desde el archivo

model_lr = joblib.load('../04_models/04_tercer_modelo.pkl')

# Definimos la función de predicción

def predict_comment(comment):
    
    labels = ['Toxic', 'Severe Toxic', 'Obscene', 'Threat', 'Insult', 'Identity Hate']
    probabilities = model_lr.predict([comment])

    # La salida de `predict` es una lista de arrays, uno por cada etiqueta

    probabilities = [prob[1] for prob in probabilities]  # Tomar la probabilidad de la clase positiva
    return dict(zip(labels, probabilities))