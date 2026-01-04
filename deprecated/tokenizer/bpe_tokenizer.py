import regex as re
import torch.nn as nn
from tokenizers import Tokenizer, models, pre_tokenizers, decoders

class BPETokenizer(nn.Module):

    def __init__(self, vocab_size, special_tokens=None):
        super().__init__()
        self.pattern = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {}
        self.special_tokens = special_tokens or ["<pad>", "<unk>", "<bos>", "<eos>", "<user>", "<assistant>"]
        self.special_token_ids = {}

    # Find Consecutive Pairs
    def get_stats(self, token_ids, stats=None):
        """Count frequency of consecutive pairs"""

        if stats is None:
            stats = {}
        for pair in zip(token_ids, token_ids[1:]):
            stats[pair] = stats.get(pair, 0) + 1

        return stats

    # Merge Token Ids
    def merge(self, token_ids, pair, new_idx):
        """Merge all occurrences of pair into new_idx"""

        new_token_ids = []
        i = 0
        while i < len(token_ids):
            if ((i + 1 < len(token_ids)) and (token_ids[i] == pair[0]) and (token_ids[i + 1] == pair[1])):
                new_token_ids.append(new_idx)
                i += 2
            else:
                new_token_ids.append(token_ids[i])
                i += 1

        return new_token_ids

    def train(self, clean_text, verbose=False):
        """Train BPE Tokenizer on clean text"""

        num_special_tokens = len(self.special_tokens)
        effective_vocab_size = self.vocab_size - num_special_tokens
        num_merges = effective_vocab_size - 256

        text_chunks = re.findall(self.pattern, clean_text)
        token_ids = [list(chunk.encode('utf-8')) for chunk in text_chunks]

        self.vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_merges):
            stats = {}

            for chunk_token in token_ids:
                self.get_stats(chunk_token, stats)

            if not stats:
                break

            top_pair = max(stats, key=stats.get)
            new_token_id = 256 + i

            if verbose:
                print(f'merged : {top_pair} -> {new_token_id }')

            token_ids = [self.merge(chunk_token, top_pair, new_token_id) for chunk_token in token_ids]
            self.merges[top_pair] = new_token_id
            self.vocab[new_token_id] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]

        self.add_special_tokens_to_vocab()

    # Adding Special Tokens to Vocabulary
    def add_special_tokens_to_vocab(self):
        
        if not self.special_tokens:  
            return

        start_id = max(self.vocab.keys()) + 1         
        for i, token in enumerate(self.special_tokens):
            token_id = start_id + i
            self.vocab[token_id] = token.encode('utf-8')
            self.special_token_ids[token] = token_id
        
        print(f"Special tokens: {self.special_token_ids}")

    def save_hf(self, path):
        """Convert to HuggingFace tokenizer and save"""

        hf_vocab = {}
        for token_id, token_bytes in self.vocab.items():
            try:
                token_str = token_bytes.decode('utf-8')
            except UnicodeDecodeError:
                token_str = ''.join(f'\\x{b:02x}' for b in token_bytes)
            hf_vocab[token_str] = token_id
        
        hf_merges = []
        for (token1_id, token2_id), merged_id in self.merges.items():
            try:
                token1_str = self.vocab[token1_id].decode('utf-8')
                token2_str = self.vocab[token2_id].decode('utf-8')
            except UnicodeDecodeError:
                token1_str = ''.join(f'\\x{b:02x}' for b in self.vocab[token1_id])
                token2_str = ''.join(f'\\x{b:02x}' for b in self.vocab[token2_id])
            
            hf_merges.append((token1_str, token2_str))
        
        tokenizer = Tokenizer(models.BPE(
            vocab=hf_vocab,
            merges=hf_merges,
            unk_token="<unk>" if "<unk>" in self.special_tokens else None
        ))
        
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(
                pattern=self.pattern,
                behavior="isolated"
            )
        ])
        tokenizer.decoder = decoders.Sequence([
            decoders.BPEDecoder()
        ])
        
        special_tokens_list = []
        for token in self.special_tokens:
            special_tokens_list.append(token)
        
        if special_tokens_list:
            tokenizer.add_special_tokens(special_tokens_list)
        
        tokenizer.save(path)
        print(f"HF tokenizer saved to {path}")
        print(f"Contains: {len(hf_vocab)} vocab, {len(hf_merges)} merges, {len(self.special_tokens)} special tokens")
        
        return tokenizer