import requests
import io
import re

import pandas as pd
import numpy as np
import streamlit as st
import torch
import gin
import comet_ml
import matplotlib.pyplot as plt
from math import ceil
from logging import getLogger

log = getLogger("test")
log.error("test")

gin.enter_interactive_mode()
from grok.data import ArithmeticTokenizer, EOS_TOKEN

torch.set_grad_enabled(False)

st.title('Grokked transformer analysis tool')

tokenizer = ArithmeticTokenizer(modulus=97)

comet_api = comet_ml.api.API()

@st.cache(ttl=60)
def get_experiments():
    return {e.name: e.key for e in comet_api.get_experiments(comet_api.get_default_workspace(), project_name="grok")
            if 'streamlit' in e.get_tags()}

def epoch_to_step(epoch):
    samples = 97*97
    batches = ceil(samples / 512)
    return epoch * batches

def on_experiment_select():
    log.error(f"Selected experiment: {st.session_state.experiment_selectbox}")
    st.session_state['experiment_selected'] = True
    experiment = comet_api.get_experiment_by_key(get_experiments()[st.session_state.experiment_selectbox])
    with st.spinner("Fetching experiment data..."):
        models = [asset for asset in experiment.get_asset_list() if asset['fileName'].startswith('transformer')]
        models = {epoch_to_step(int(m['fileName'].split('_')[1].split('.')[0])): m for m in models}
        train_acc = [(m['step'], float(m['metricValue'])) for m in experiment.get_metrics(metric='train_accuracy')]
        val_acc = [(m['step'], float(m['metricValue'])) for m in experiment.get_metrics(metric='val_accuracy')]
        metrics = train_acc, val_acc
        st.session_state['metrics'] = metrics
        st.session_state['models'] = models

col1, col2 = st.columns(2)
col1.selectbox("",
             get_experiments().keys(),
             key='experiment_selectbox',
             index=0)
if col2.button('Select experiment'):
    on_experiment_select()

@st.cache()
def get_model(model_link):
    response = requests.get(model_link)
    return torch.load(io.BytesIO(response.content), map_location=torch.device('cpu'))

def get_predictions(model, inputs):
    input_tokens = tokenizer.encode(inputs)
    output, _, _ = model(input_tokens)
    output_tokens = output.argmax(axis=2)
    result_strings = [tokenizer.decode(o) for o in output_tokens]
    return [s.split()[-1] for s in result_strings]

def on_checkpoint_select():
    log.error(f"Selected checkpoint: {st.session_state.checkpoint_select_slider}")
    with st.spinner("Evaluating checkpoint..."):
        model = get_model(st.session_state.models[st.session_state.checkpoint_select_slider]['link'])
        failure_cases = []
        progress_bar = st.progress(0.)
        for x in range(97):
            queries = [f"{EOS_TOKEN} {x} + {y} =" for y in range(97)]
            predictions = get_predictions(model, queries)
            for y in range(97):
                prediction = predictions[y]
                expected = (x + y) % 97
                if int(prediction) != expected:
                    failure_cases.append((x, y, prediction, expected))
                progress_bar.progress((x*97 + y + 1)/(97 * 97))
        progress_bar.empty()
        failure_cases = pd.DataFrame(failure_cases, columns=('x', 'y', 'predicted', 'expected'))

    st.write(f"{failure_cases.shape[0]} failure cases found (of {97*97} cases)")
    st.dataframe(failure_cases)
    correct_map = np.zeros((97, 97))
    for x, y in failure_cases[['x', 'y']].to_numpy():
        correct_map[x, y] = 1
    fig, ax = plt.subplots()
    ax.imshow(correct_map, origin='lower')
    st.pyplot(fig)


if 'experiment_selected' in st.session_state:

    fig, ax = plt.subplots()
    ax.plot(*zip(*sorted(st.session_state.metrics[0])), label='train_acc')
    ax.plot(*zip(*sorted(st.session_state.metrics[1])), label='val_acc')
    ax.legend()
    ax.set_xlabel('step')
    if 'checkpoint_select_slider' in st.session_state:
        ax.axvline(st.session_state.checkpoint_select_slider)
    ax.set_xscale('log')
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    col1.select_slider("Select checkpoint",
                       options=sorted(st.session_state.models.keys()),
                       key='checkpoint_select_slider',
                       format_func=lambda x: f"{x:.2e}")
    if col2.button("Evaluate checkpoint"):
        on_checkpoint_select()
