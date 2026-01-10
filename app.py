import streamlit as st
import torch
from tokenizers import Tokenizer
from model.neuroformer import NeuroFormer

# --------------------------------------------------
# Streamlit config
# --------------------------------------------------
st.set_page_config(
    page_title="NeuroFormer Playground",
    layout="wide"
)

# --------------------------------------------------
# Constants
# --------------------------------------------------
MODEL_PATH = "training/best_model.pt"
TOKENIZER_PATH = "tokenizer/bpe_tokenizer.json"

VOCAB_SIZE = 1000
SEQ_LEN = 512
DEVICE = "cpu"

# --------------------------------------------------
# Caching
# --------------------------------------------------
@st.cache_resource
def load_tokenizer(path):
    return Tokenizer.from_file(path)

@st.cache_resource
def load_model(path, vocab_size, seq_len):
    model = NeuroFormer(
        vocab_size=vocab_size,
        sequence_length=seq_len
    )
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# --------------------------------------------------
# Load resources
# --------------------------------------------------
tokenizer = load_tokenizer(TOKENIZER_PATH)
model = load_model(MODEL_PATH, VOCAB_SIZE, SEQ_LEN)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("NeuroFormer")

page = st.sidebar.selectbox(
    "Choose Mode",
    ["Script Generator"]
)

st.sidebar.subheader("Generation Settings")
max_new_tokens = st.sidebar.slider("Max New Tokens", 50, 500, 200)
temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
top_k = st.sidebar.slider("Top-K", 1, 100, 50)
top_p = st.sidebar.slider("Top-P", 0.0, 1.0, 0.9, 0.05)

# --------------------------------------------------
# Generation function
# --------------------------------------------------
@torch.no_grad()
def generate_text(model, tokenizer, prompt):
    # Add BOS token explicitly 
    if not prompt.startswith("<BOS>"):
        prompt = "<BOS> " + prompt

    input_ids = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)

    output_ids = model.generate(
        input_ids=input_ids,
        max_length=input_ids.shape[1] + max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    return tokenizer.decode(output_ids[0].tolist())

# --------------------------------------------------
# UI
# --------------------------------------------------
if page == "Script Generator":
    st.header("🧠 NeuroFormer Script Generator")

    prompt = st.text_area(
        "Enter a prompt",
        value="Hello, how are you today?",
        height=120
    )

    if st.button("Generate"):
        with st.spinner("Generating..."):
            text = generate_text(model, tokenizer, prompt)
        st.success(text)