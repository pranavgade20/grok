import requests
import io
import re

import streamlit as st
import torch
import gin
import comet_ml

gin.enter_interactive_mode()
from grok.data import ArithmeticDataset, ArithmeticTokenizer, EOS_TOKEN

torch.set_grad_enabled(False)

st.title('Grokked transformer analysis tool')

comet_api = comet_ml.api.API()
with st.spinner("Fetching available experiments..."):
    experiments = comet_api.get_experiments(comet_api.get_default_workspace(), project_name="grok")

    experiment = st.selectbox("Select comet.ml experiment",
                              ['select...'] + experiments,
                              format_func=lambda e: e if isinstance(e, str) else e.get_name())

if experiment == 'select...':
    st.write("please select experiment")
    st.stop()


with st.spinner("Fetching available checkpoints..."):
    models = [asset for asset in experiment.get_asset_list() if asset['fileName'].startswith('transformer')]

    selected_model = st.selectbox("Select model checkpoint",
                                  ["select..."]+models,
                                  format_func=lambda m: m if isinstance(m, str) else m['fileName'],
                                  disabled=(experiment == "select..."))

@st.cache()
def get_model(model_link):
    response = requests.get(model_link)
    return torch.load(io.BytesIO(response.content), map_location=torch.device('cpu'))

def get_prediction(model, input_string):
    input_tokens = tokenizer.encode(f"{EOS_TOKEN} {input_string} {EOS_TOKEN}")
    output, _, _ = model(input_tokens[:-1])
    result = tokenizer.decode(output.argmax(axis=1)).split()[-1]
    return int(result)

if selected_model == 'select...':
    st.write("please select model")
    st.stop()

model = get_model(selected_model['link'])
tokenizer = ArithmeticTokenizer(modulus=97)
query = st.text_input("Put the query in format 'x + y ='")

if not re.search(r"^\d+ \+ \d+ =$", query):
    st.write(f"incorrect query {query}")
    st.stop()

st.write(f"{query} {get_prediction(model, query)}")
