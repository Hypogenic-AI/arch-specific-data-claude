# Datasets for Architecture-Specific Data Attractors Research

Data files are NOT committed to git due to size. Follow download instructions below.

## Dataset 1: TinyStories

### Overview
- **Source**: HuggingFace `roneneldan/TinyStories`
- **Size**: ~530K stories, ~249 MB (parquet)
- **Format**: Parquet (single text column)
- **Task**: Language modeling / text generation
- **License**: MIT
- **Why relevant**: Small, simple stories ideal for training small RNNs and transformers from scratch. The controlled vocabulary and narrative structure makes it easier to observe attractor formation.

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("roneneldan/TinyStories", split="train")
dataset.save_to_disk("datasets/tinystories/data")
```

Or download parquet directly:
```bash
curl -L "https://huggingface.co/api/datasets/roneneldan/TinyStories/parquet/default/train/0.parquet" \
  -o datasets/tinystories/train_0.parquet
```

### Loading
```python
import pyarrow.parquet as pq
table = pq.read_table("datasets/tinystories/train_0.parquet")
texts = table.column("text").to_pylist()
```

---

## Dataset 2: WikiText-2

### Overview
- **Source**: HuggingFace `Salesforce/wikitext` (wikitext-2-raw-v1)
- **Size**: ~2M tokens, ~6.4 MB (parquet)
- **Format**: Parquet (single text column)
- **Task**: Language modeling
- **Splits**: train, validation, test
- **License**: Creative Commons Attribution-ShareAlike
- **Why relevant**: Standard language modeling benchmark. Good for comparing how architectures learn to model natural language and what default patterns they converge to.

### Download Instructions

```bash
curl -L "https://huggingface.co/api/datasets/Salesforce/wikitext/parquet/wikitext-2-raw-v1/train/0.parquet" \
  -o datasets/wikitext2/train.parquet
```

### Loading
```python
import pyarrow.parquet as pq
table = pq.read_table("datasets/wikitext2/train.parquet")
texts = table.column("text").to_pylist()
```

---

## Dataset 3: Synthetic Preference Questions

### Overview
- **Source**: Custom-generated for this research
- **Size**: 10 questions (expandable)
- **Format**: JSON
- **Task**: Testing default/attractor responses across architectures
- **Why relevant**: Directly tests the research hypothesis - do different architectures converge to different "favorite" answers when asked subjective questions?

### Loading
```python
import json
with open("datasets/synthetic_preferences/questions.json") as f:
    questions = json.load(f)
```

---

## Dataset 4: MAGE (recommended, not downloaded)

### Overview
- **Source**: Used in the Attractor Cycles paper (2502.15208)
- **Size**: 1000 sentences + 300 paragraphs
- **Task**: Successive paraphrasing experiments
- **Why relevant**: The exact dataset used to demonstrate 2-period attractor cycles in LLMs

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("yaful/MAGE")
```

---

## Dataset 5: PersonaChat (recommended, not downloaded)

### Overview
- **Source**: HuggingFace `bavard/personachat_truecased`
- **Size**: ~131K dialogue turns
- **Task**: Persona-conditioned dialogue
- **Why relevant**: Tests whether different architectures develop different "personas" or default conversational styles

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("facebook/personachat", split="train")
```

---

## Recommended Usage for Experiments

1. **Attractor formation comparison**: Train small RNN and Transformer on TinyStories/WikiText-2, compare internal representations at each layer/timestep
2. **Default response testing**: Use preference questions to probe what "defaults" each architecture develops
3. **Successive transformation**: Replicate attractor cycles experiment from 2502.15208 but compare transformer vs RNN
4. **Persona stability**: Use PersonaChat to test whether architectures maintain or collapse personas differently
