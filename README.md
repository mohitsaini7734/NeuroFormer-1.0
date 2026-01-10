# NeuroFormer 2.0 

NeuroFormer 2.0 is a **subword-level Transformer Architecture** built from scratch in **PyTorch**.  
It includes a custom **Byte Pair Encoding (BPE) tokenizer**, a manually implemented **Transformer architecture**, and a **Streamlit UI**.  
The model is trained on **Cornell Movie-Dialog Corpus** to learn conversational patterns.

---

## Features
- **Custom BPE Tokenizer** – trained from scratch on Shakespeare dataset  
- **Transformer architecture** – self-attention, feed-forward, positional encoding, layer normalization   
- **Streamlit UI** – user-friendly interface for chatting with the model  
- **Fully modular design** – easy to extend, with cleaner code practices  

---

## Streamlit UI
```
https://neuroformer.streamlit.app/
```

---

## Training Summary

### Tokenizer
- **Type:** Byte Pair Encoding (BPE)
- **Vocabulary size:** ~1,000 subword tokens
- **Training corpus:** Dialogue dataset + conversational text

### Model
- **Architecture:** Decoder-only Transformer
- **Sequence length:** 512
- **Training type:** From scratch
- **Final validation loss:** **2.43**
- **Perplexity:** ~**11.4**

### Dataset
- **Primary dataset:** Dialogue-style conversational corpus  
- **Focus:** Natural language flow, turn-based conversation, coherence  
- **Train/validation split:** Custom curated split  

> Earlier Shakespeare-based experiments are preserved for reference and moved to `deprecated/`.

---

## Project Structure
```


NeuroFormer2.0/
│
├── data/                           # Datasets + preprocessing
│ ├── dialogues.txt                 # Raw Cornell dataset
│ ├── train.txt                     # Train split for Cornell model
│ ├── val.txt                       # Validation split for Cornell model
│ ├── dataloader.py                 # Data loading utilities
│ └── DatasetPreprocessing.py       # Dataset cleaning & preprocessing
│
├── deprecated/                     # Older experiments & code 
|
├── model/                          # Transformer model implementation
│ ├── neuroformer.py                # Custom Transformer model
│ └── NeuroFormer.ipynb             # Jupyter notebook for experiments
│
├── tokenizer/                      # Custom BPE Tokenizer
│ ├── tokenization.py               # Script to train BPE tokenizer
│ ├── BPE_Tokenizer.ipynb           # Notebook for tokenizer experiments
│ ├── bpe_tokenizer.json            # Saved vocab from BPE tokenizer
│ ├── bpe_tokenizer.py              # Custom BPE tokenizer implementation
│ └── init.py
│
├── training/                       # Training 
│ ├── cornell-model-training.ipynb  # Training Cornell Script generator
│ ├── best_model.pt                 # Best Model weights
│
├── app.py                          # Streamlit chatbot UI
├── requirements.txt                # Project dependencies
├── README.md                       # Documentation
└── .gitignore / .gitattributes

```

---

## Future Improvements

- Increase BPE vocab size → improve handling of rare words.
- Experiment with new attention mechanisms.
- Longer training schedule for stability & better convergence.
---

## Acknowledgements  

- This project is inspired by the work of **[Andrej Karpathy](https://www.youtube.com/@AndrejKarpathy)**, **[Sebastian Raschka](https://www.youtube.com/@sebastianraschka)**, and the open-source ML community.

---
