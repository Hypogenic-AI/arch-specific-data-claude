# Cloned Code Repositories

## 1. rnn-icrag
- **URL**: https://github.com/dangxingyu/rnn-icrag
- **Purpose**: Code for "RNNs Are Not Transformers (Yet)" (ICLR 2025). Compares RNN (Mamba) and Transformer (LLaMA) on algorithmic tasks.
- **Location**: `code/rnn-icrag/`
- **Key files**: `rnn/` directory contains model implementations, `configs/` has experiment configs
- **Relevance**: Provides RNN vs Transformer comparison framework. Can be adapted to test attractor dynamics differences between architectures.
- **Notes**: Uses Mamba as RNN baseline, LLaMA 2 architecture for transformer. Tests on IsTree and HotPot-QA.

## 2. nanoGPT
- **URL**: https://github.com/karpathy/nanoGPT
- **Purpose**: Minimal GPT training framework. Simple, readable implementation for training small transformers from scratch.
- **Location**: `code/nanoGPT/`
- **Key files**: `model.py` (GPT model), `train.py` (training script), `data/` (data preparation)
- **Relevance**: Ideal baseline for training small transformers on our datasets. Can extract intermediate representations for attractor analysis.
- **Notes**: Supports training on custom text data. Well-documented and widely used.

## 3. minGRU-pytorch
- **URL**: https://github.com/lucidrains/minGRU-pytorch
- **Purpose**: Minimal GRU implementation in PyTorch. Recent efficient RNN design.
- **Location**: `code/minGRU-pytorch/`
- **Key files**: `minGRU_pytorch/` (model code), `train.py` (training), `train_hybrid.py` (hybrid training)
- **Relevance**: Modern, efficient RNN implementation that can be paired with nanoGPT for architecture comparison experiments. Includes hybrid RNN-Transformer training.
- **Notes**: Has both pure RNN and hybrid (RNN + attention) variants, useful for testing the spectrum between architectures.

## 4. RENN (Reverse Engineering Neural Networks) - Google Research
- **URL**: https://github.com/google-research/reverse-engineering-neural-networks
- **Purpose**: JAX-based toolkit for analyzing RNN dynamics. Provides tools for finding and analyzing approximate fixed points of trained RNNs.
- **Location**: `code/renn/`
- **Key files**: `renn/` (library), `notebooks/` (analysis examples), `scripts/` (training)
- **Relevance**: **Critical for RNN attractor analysis.** Provides fixed-point finding, linearized dynamics, and attractor characterization tools. Based on Maheswaranathan et al. (NeurIPS 2019) "line attractor dynamics" work.
- **Notes**: JAX-based (not PyTorch). May need adaptation for our PyTorch models, but the methodology is directly applicable.

## 5. PyTorch Word Language Model (RNN + Transformer in one codebase)
- **URL**: https://github.com/pytorch/examples/tree/main/word_language_model
- **Purpose**: Official PyTorch example training RNN (Elman, GRU, LSTM) OR Transformer on WikiText-2. Single `main.py`, model selected via `--model` flag.
- **Location**: `code/pytorch-examples/word_language_model/`
- **Key files**: `main.py` (training), `model.py` (both architectures), `data.py` (data loading)
- **Relevance**: **Critical starting point.** Same data pipeline, same training loop, different architectures. Can be extended with attractor analysis hooks.
- **Notes**: Minimal and self-contained. Perfect for controlled RNN vs Transformer comparison.

## Recommended Experiment Approach

1. Use **nanoGPT** to train a small transformer on TinyStories/WikiText-2
2. Use **minGRU-pytorch** to train a comparable RNN on the same data
3. Extract hidden states from both models at each layer/timestep
4. Analyze attractor formation: do representations converge to concept-specific regions?
5. Compare attractor structure between architectures
6. Use preference questions to test default response patterns
7. Adapt **rnn-icrag** framework for more rigorous RNN vs Transformer comparison
