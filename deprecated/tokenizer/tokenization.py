from bpe_tokenizer import BPETokenizer
from tokenizers import Tokenizer
import regex as re

special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>", "<user>", "<assistant>"]
tokenizer = BPETokenizer(vocab_size=2000, special_tokens=special_tokens)

with open("Data/shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()
    
text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]+", " ", text)  
text = re.sub(r"\n\s*\n+", "\n\n", text)  
text = text.strip()

tokenizer.train(text, verbose=True)
tokenizer.save_hf("Tokenizer/bpe_tokenizer.json")

hf_tokenizer = Tokenizer.from_file("Tokenizer/bpe_tokenizer.json")

test_text = "<bos>Hi I am Nishchal, a student from IIIT Lucknow.<eos>"
encoded = hf_tokenizer.encode(test_text)
decoded = hf_tokenizer.decode(encoded.ids)

print("\nTest Results:")
print("Original:", test_text)
print("Encoded:", encoded.ids)
print("Decoded:", decoded)
print("Vocab size:", hf_tokenizer.get_vocab_size())