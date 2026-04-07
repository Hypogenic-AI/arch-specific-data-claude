"""
Data loading and tokenization for TinyStories dataset.
Uses a simple character-level or BPE tokenizer for fair comparison.
"""

import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
import pyarrow.parquet as pq


class SimpleTokenizer:
    """Simple word-level tokenizer with fixed vocabulary."""

    def __init__(self, texts, max_vocab=8192):
        # Count word frequencies
        from collections import Counter
        word_counts = Counter()
        for text in texts:
            word_counts.update(text.lower().split())

        # Build vocabulary from most common words
        special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
        vocab_words = [w for w, _ in word_counts.most_common(max_vocab - len(special_tokens))]
        self.vocab = special_tokens + vocab_words
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2word = {i: w for i, w in enumerate(self.vocab)}
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3
        self.vocab_size = len(self.vocab)

    def encode(self, text, max_len=256):
        words = text.lower().split()
        ids = [self.bos_id]
        for w in words[:max_len - 2]:
            ids.append(self.word2idx.get(w, self.unk_id))
        ids.append(self.eos_id)
        return ids

    def decode(self, ids):
        words = []
        for idx in ids:
            if idx == self.eos_id:
                break
            if idx in (self.pad_id, self.bos_id):
                continue
            words.append(self.idx2word.get(idx, "<unk>"))
        return " ".join(words)

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"vocab": self.vocab}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        tok = cls.__new__(cls)
        tok.vocab = data["vocab"]
        tok.word2idx = {w: i for i, w in enumerate(tok.vocab)}
        tok.idx2word = {i: w for i, w in enumerate(tok.vocab)}
        tok.pad_id = 0
        tok.unk_id = 1
        tok.bos_id = 2
        tok.eos_id = 3
        tok.vocab_size = len(tok.vocab)
        return tok


class TextDataset(Dataset):
    """Dataset that returns non-overlapping fixed-length token sequences for LM training."""

    def __init__(self, token_ids, seq_len=128):
        self.seq_len = seq_len
        # Flatten all token IDs into one long sequence
        data = torch.tensor(token_ids, dtype=torch.long)
        # Non-overlapping chunks for efficiency
        n_chunks = len(data) // (seq_len + 1)
        self.data = data[:n_chunks * (seq_len + 1)].view(n_chunks, seq_len + 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        return chunk[:-1], chunk[1:]


def load_tinystories(data_dir="datasets/tinystories", max_stories=50000):
    """Load TinyStories from parquet file."""
    parquet_file = os.path.join(data_dir, "train_0.parquet")
    print(f"Loading TinyStories from {parquet_file}...")

    table = pq.read_table(parquet_file)
    df = table.to_pandas()

    # Get text column
    text_col = df.columns[0]  # Usually 'text'
    texts = df[text_col].tolist()[:max_stories]
    print(f"Loaded {len(texts)} stories")

    return texts


def prepare_data(texts, vocab_size=8192, seq_len=128, val_split=0.05):
    """Tokenize texts and create train/val datasets."""
    print("Building tokenizer...")
    tokenizer = SimpleTokenizer(texts, max_vocab=vocab_size)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    print("Tokenizing...")
    all_ids = []
    for text in texts:
        ids = tokenizer.encode(text, max_len=512)
        all_ids.extend(ids)

    print(f"Total tokens: {len(all_ids):,}")

    # Split
    split_idx = int(len(all_ids) * (1 - val_split))
    train_ids = all_ids[:split_idx]
    val_ids = all_ids[split_idx:]

    train_dataset = TextDataset(train_ids, seq_len=seq_len)
    val_dataset = TextDataset(val_ids, seq_len=seq_len)

    print(f"Train sequences: {len(train_dataset):,}")
    print(f"Val sequences: {len(val_dataset):,}")

    return tokenizer, train_dataset, val_dataset
