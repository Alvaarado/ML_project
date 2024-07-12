# %% [markdown]
# ### Cargar

# %%
import pickle
import tensorflow as tf
import numpy as np
import pandas as pd

# Cargar el diccionario del archivo .pkl
with open('./04_models/02_modelo_1_and_objects.pkl', 'rb') as f:
    saved_objects = pickle.load(f)

model_path = saved_objects['model_path']
vectorizer = saved_objects['vectorizer']
precision_value = saved_objects['precision']
recall_value = saved_objects['recall']

# %%

# Cargar el modelo desde la ruta guardada
model = tf.keras.models.load_model('./04_models/01_primer_modelo.keras')

# %%
input_text_2 = vectorizer('You freaking suck! I am going to kill you')

# %%
predic = model.predict(np.expand_dims(input_text_2,0))
print(predic)
