import gradio as gr
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

with open('../04_models/02_modelo_1_and_objects.pkl', 'rb') as f:
    saved_objects = pickle.load(f)

df = pd.read_csv('../02_processed/comments_df.csv')

model = tf.keras.models.load_model('../04_models/01_primer_modelo.keras')
model_path = saved_objects['model_path']
vectorizer = saved_objects['vectorizer']
precision_value = saved_objects['precision']
recall_value = saved_objects['recall']

input_str = vectorizer()
res = model.predict(np.expand_dims(input_str,0))
res > 0.5

def score_comment (comment):
    vectorized_comment = vectorizer([comment])
    results = model.predict(vectorized_comment)
    text = ''
    for idx, col in enumerate(df.columns[2:]):
        text += '{}: {}\n'.format(col, results[0][idx]>0.5)
    
    return text

interface = gr.Interface(fn=score_comment, inputs= gr.inputs.Textbox(lines=2,placeholder='Comment to score'),
                         outputs='text')
interface.launch(share=True)