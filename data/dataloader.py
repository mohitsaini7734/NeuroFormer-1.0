import torch
from torch.utils.data import Dataset, DataLoader

class NeuroFormerDataset(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text).ids

        assert len(token_ids) > max_length, f"Number of tokenized inputs ({len(token_ids)}) must be greater than max_length ({max_length})"

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def load_data(text, tokenizer, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True):

    dataset = NeuroFormerDataset(text, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )

    return dataloader