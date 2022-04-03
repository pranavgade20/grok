import requests
import io
import re

import streamlit as st
import torch
import gin
import comet_ml
import matplotlib.pyplot as plt
from math import ceil

gin.enter_interactive_mode()
from grok.data import ArithmeticTokenizer, EOS_TOKEN

torch.set_grad_enabled(False)

st.title('Grokked transformer analysis tool')

tokenizer = ArithmeticTokenizer(modulus=97)

comet_api = comet_ml.api.API()
@st.cache(ttl=120, allow_output_mutation=True)
def get_experiments():
    return tuple(e for e in comet_api.get_experiments(comet_api.get_default_workspace(), project_name="grok")
            if 'streamlit' in e.get_tags())

def epoch_to_step(epoch):
    samples = 97*97
    batches = ceil(samples / 512)
    return epoch * batches

def on_experiment_select():
    st.session_state['experiment_selected'] = True
    experiment = st.session_state.experiment_selectbox
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
             get_experiments(),
             key='experiment_selectbox',
             format_func=lambda e: e.get_name())
if col2.button('Select experiment'):
    on_experiment_select()

@st.cache()
def get_model(model_link):
    response = requests.get(model_link)
    return torch.load(io.BytesIO(response.content), map_location=torch.device('cpu'))

def get_prediction(model, input_string):
    input_tokens = tokenizer.encode(input_string)
    output, _, _ = model(input_tokens)
    result = tokenizer.decode(output.argmax(axis=1)).split()[-1]
    return int(result)

def on_checkpoint_select():
    with st.spinner("Evaluating checkpoint..."):
        model = get_model(st.session_state.models[st.session_state.checkpoint_select_slider]['link'])
        failure_cases = []
        for x in range(97):
            for y in range(97):
                query = f"{EOS_TOKEN} {x} + {y} ="
                prediction = get_prediction(model, query)
                expected = (x + y) % 97
                if prediction != expected:
                    failure_cases.append(f"{x} + {y} = {prediction} (expected {expected})")
    st.write(failure_cases)


if 'experiment_selected' in st.session_state:

    fig, ax = plt.subplots()
    ax.plot(*zip(*sorted(st.session_state.metrics[0])))
    ax.plot(*zip(*sorted(st.session_state.metrics[1])))
    ax.set_xscale('log')
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    col1.select_slider("Select checkpoint",
                       options=st.session_state.models.keys(),
                       key='checkpoint_select_slider')
    if col2.button("Evaluate checkpoint"):
        on_checkpoint_select()
