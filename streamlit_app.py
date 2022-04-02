import streamlit as st
import torch
import gin
gin.enter_interactive_mode()

from grok.data import ArithmeticDataset, ArithmeticTokenizer, EOS_TOKEN

torch.set_grad_enabled(False)

tokenizer = ArithmeticTokenizer(modulus=97)

def query_model(input_string):
    input_tokens = tokenizer.encode(f"{EOS_TOKEN} {input_string} {EOS_TOKEN}")
    output, _, _ = get_model()(input_tokens[:-1])
    result = tokenizer.decode(output.argmax(axis=1)).split()[-1]
    return int(result)

@st.cache()
def get_model():
    return torch.load('transformer_30000.pt', map_location=torch.device('cpu'))

st.title('Grokked transformer analysis tool')

query = st.text_input("Put the query in format 'x + y ='")
st.write(f"{query} {query_model(query)}")
