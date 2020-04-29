import os
import sys
sys.path.append(".")
import json
import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st

from text_classification import config
from text_classification import predict
from text_classification import utils


def softmax(x):
    x = x - max(x)
    y = np.exp(x)
    return y / sum(y)


def normalize(x):
    return (x-min(x)) / (max(x)-min(x))


# Title
st.title("Made With ML · Creating an End-to-End ML Application")
st.write("""[<img src="https://github.com/madewithml/images/blob/master/images/yt.png?raw=true" style="width:1.2rem;"> Watch Lesson](https://www.youtube.com/channel/UCaVCnFQXS7PYMoYZu3KdC0Q/featured) · [<img src="https://github.com/madewithml/images/blob/master/images/github_logo.png?raw=true" style="width:1.1rem;"> GitHub](https://www.github.com/madewithml/lessons) · [<img src="https://avatars0.githubusercontent.com/u/60439358?s=200&v=4" style="width:1.2rem;"> Made With ML](https://madewithml.com)""", unsafe_allow_html=True)

# Pages
page = st.sidebar.selectbox(
    "Choose a page", ['Inference', 'Model details'])
if page == 'Inference':

    st.header("🚀 Try it out!")

    # Input text
    text = st.text_input(
        "Enter text to classify", value="The Canadian minister signed in the new federal law.")
    results = predict.predict(experiment_id='latest',
                              inputs=[{'text': text}])

    # Results
    raw_text = results[0]['raw_input']
    st.write("**Raw text**:", raw_text)
    preprocessed_text = results[0]['preprocessed_input']
    st.write("**Preproessed text**:", preprocessed_text)
    st.write("**Probabilities**:")
    st.json(results[0]['probabilities'])

    # Interpretability
    st.write("**Top n-grams**:")
    words = preprocessed_text.split(' ')
    top_n_grams = {}
    token_index_to_freq = {i: 0 for i in range(len(words))}
    for filter_size, d in results[0]['top_n_grams'].items():
        top_n_grams[filter_size] = d['n_gram']
        start = int(d['start'])
        end = int(d['end'])
        for i in range(start, end):
            token_index_to_freq[i] += 1
    softmax_values = softmax(np.fromiter(token_index_to_freq.values(), dtype=int))
    normalized_values = normalize(x=softmax_values)

    # Heatmap
    html = "<div>"
    for i, token in enumerate(words):
        html += f'<span style="background-color: hsl(100, 100%, {((1-normalized_values[i]) * 53 + 50)}%">' + \
            token + ' </span>'
    html += "</div><br>"
    st.write(html, unsafe_allow_html=True)
    st.json(top_n_grams)

    # Warning
    st.info("""⚠️The model architecture used in this demo is **not state-of-the-art** and it's not **fully optimized**, as this was not the main focus of the lesson.
            Also keep in mind that the **dataset** is dated and **limited** to particular vocabulary.
            If you are interested in generalized text classification or NLP in general, check out these [curated resources](https://madewithml.com/topics/#nlp).""")

    # Show raw json
    show_json = st.checkbox("Show complete JSON output:", value=False)
    if show_json:
        st.json(results)


elif page == 'Model details':

    st.header("Model details")

    # Get experiment
    experiment_id = max(os.listdir(config.EXPERIMENTS_DIR))
    experiment_dir = os.path.join(config.EXPERIMENTS_DIR, experiment_id)
    st.write("**ID**:", experiment_id)

    # Performance
    st.write("**Performance**:")
    performance = utils.load_json(
        os.path.join(experiment_dir, 'performance.json'))
    st.json(performance)

    # Confusion matrix
    st.image(os.path.join(experiment_dir, 'confusion_matrix.png'))

    # Config
    st.write("**Config**:")
    experiment_config = utils.load_json(
        os.path.join(experiment_dir, 'config.json'))
    st.json(experiment_config)
