from bpe_tokenizer import BPETokenizer
from tokenizers import Tokenizer
import regex as re

# MUST match dataset exactly
special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

tokenizer = BPETokenizer(
    vocab_size=1000,
    special_tokens=special_tokens
)

# --------------------------------------------------
# Load and clean text data
# --------------------------------------------------
with open("Data/dialogues.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Minimal cleaning only
text = re.sub(r"\n\s*\n+", "\n\n", text)
text = text.strip()

# --------------------------------------------------
# Train tokenizer
# --------------------------------------------------
tokenizer.train(text, verbose=True)

tokenizer.save_hf("Tokenizer/bpe_tokenizer.json")

# --------------------------------------------------
# Load HF tokenizer for testing
# --------------------------------------------------
hf_tokenizer = Tokenizer.from_file("Tokenizer/bpe_tokenizer.json")

# --------------------------------------------------
# Test on REAL dialogue-style input
# --------------------------------------------------
test_text = (
    "<BOS>\n"
    "<CHAR_BIANCA> Hi, I am Nishchal.\n"
    "<CHAR_CAMERON> Nice to meet you.\n"
    "<EOS>"
)

encoded = hf_tokenizer.encode(test_text)
decoded = hf_tokenizer.decode(encoded.ids)

print("\nTest Results:")
print("Original:", test_text)
print("Encoded:", encoded.ids)
print("Decoded:", decoded)
print("Vocab size:", hf_tokenizer.get_vocab_size())