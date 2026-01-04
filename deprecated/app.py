import streamlit as st
import torch
from model.neuroformer import NeuroFormer   
from tokenizers import Tokenizer  

@st.cache_resource
def load_tokenizer(tokenizer_path):    
    return Tokenizer.from_file(tokenizer_path)

@st.cache_resource
def load_model(weights_path, vocab_size, sequence_length):
    model = NeuroFormer(vocab_size=vocab_size, sequence_length=sequence_length)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model

SHAKESPEARE_MODEL_PATH = "training/shakespeare.pt"
CHATBOT_MODEL_PATH = "training/chatbot.pt"
TOKENIZER = "tokenizer/bpe_tokenizer.json"

VOCAB_SIZE = 2000
SEQ_LEN = 512

shakespeare_model = load_model(SHAKESPEARE_MODEL_PATH, vocab_size=VOCAB_SIZE, sequence_length=SEQ_LEN)
chatbot_model = load_model(CHATBOT_MODEL_PATH, vocab_size=VOCAB_SIZE, sequence_length=SEQ_LEN)
tokenizer = load_tokenizer(TOKENIZER)

st.set_page_config(page_title="NeuroFormer Playground", layout="wide")
st.sidebar.title("NeuroFormer")
page = st.sidebar.radio("Choose a model:", ["Shakespeare Generator", "Chatbot"])

st.sidebar.subheader("Generation Settings")
max_length = st.sidebar.slider("Max Length", 50, 500, 200)
temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
top_k = st.sidebar.slider("Top-K", 1, 100, 50)
top_p = st.sidebar.slider("Top-P", 0.0, 1.0, 0.9, 0.05)

def generate_text(model, tokenizer, prompt):
    input_ids = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    output_ids = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    return tokenizer.decode(output_ids[0].tolist())

if page == "Shakespeare Generator":
    st.header("Shakespeare Text Generator")
    prompt = st.text_area("Enter a prompt:", "Once Upon a midnight dreary, while I pondered, weak and weary,")
    if st.button("Generate Shakespeare Text"):
        generated_text = generate_text(shakespeare_model, tokenizer, prompt)
        st.success(generated_text)

elif page == "Chatbot":
    st.header("NeuroFormer Chatbot")
    user_input = st.text_area("You:", "Hello, how are you?")
    if st.button("Get Response"):
        response = generate_text(chatbot_model, tokenizer, user_input)
        st.success(response)