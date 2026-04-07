# Resources Catalog

## Summary
This document catalogs all resources gathered for the Architecture-Specific Data Attractors research project, including papers, datasets, and code repositories.

## Papers
Total papers downloaded: 13

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Concept Attractors in LLMs | Chytas, Singh | 2025 | papers/2601.11575_concept_attractors_llms.pdf | IFS framework for LLM attractors |
| Attractor Cycles in LLMs | Wang, Li et al. | 2025 | papers/2502.15208_attractor_cycles_paraphrasing.pdf | 2-period cycles in paraphrasing |
| Transformer vs RNN Attractor Dynamics | Fukushima, Tani | 2023 | papers/2311.10763_transformer_vs_rnn_attractor_dynamics.pdf | **Most relevant** - direct comparison |
| RNNs Are Not Transformers (Yet) | Wen, Dang, Lyu | 2024 | papers/2402.18510_rnns_not_transformers_yet_iclr2025.pdf | Representation gap proof |
| Transformers are Multi-State RNNs | Various | 2024 | papers/2401.06104_transformers_multistate_rnns.pdf | Architecture bridging |
| When Transformers Outperform RNNs | Various | 2025 | papers/2503.11272_when_transformers_outperform_rnns.pdf | Statistical analysis |
| On RNNs and Transformers | Various | 2023 | papers/2301.03044_rnns_not_transformers.pdf | Earlier comparison |
| Verbalized Sampling (Mode Collapse) | Various | 2025 | papers/2510.01171_verbalized_sampling_mode_collapse.pdf | Mode collapse mitigation |
| Sycophancy Origins in LLMs | Various | 2025 | papers/2508.02087_sycophancy_origins_llms.pdf | Default behavior patterns |
| Model Collapse Demystified | Various | 2024 | papers/2402.07712_model_collapse_regression.pdf | Collapse theory |
| Interpreting RNN Excitable Attractors | Various | 2018 | papers/1807.10478_interpreting_rnn_excitable_attractors.pdf | RNN attractor extraction |
| Fading Memory in Residual RNNs | Various | 2022 | papers/2212.03771_fading_memory_inductive_bias_rnn.pdf | RNN inductive bias |
| Convergence of Training Transformers | Various | 2024 | papers/2409.17335_convergence_training_transformers.pdf | Training theory |

See `papers/README.md` for detailed descriptions.

## Datasets
Total datasets downloaded: 3 (+ 2 recommended)

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| TinyStories | HuggingFace | 530K stories, 249MB | Language modeling | datasets/tinystories/ | Downloaded (parquet) |
| WikiText-2 | HuggingFace/Salesforce | 2M tokens, 6.4MB | Language modeling | datasets/wikitext2/ | Downloaded (parquet) |
| Synthetic Preferences | Custom | 10 questions | Default response testing | datasets/synthetic_preferences/ | Created |
| MAGE | HuggingFace yaful/MAGE | 1K sentences | Paraphrasing | Not downloaded | Recommended |
| PersonaChat | Facebook | 131K turns | Dialogue | Not downloaded | Recommended |

See `datasets/README.md` for detailed descriptions and download instructions.

## Code Repositories
Total repositories cloned: 5

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| rnn-icrag | github.com/dangxingyu/rnn-icrag | RNN vs Transformer comparison (ICLR 2025) | code/rnn-icrag/ | Mamba vs LLaMA |
| nanoGPT | github.com/karpathy/nanoGPT | Small transformer training | code/nanoGPT/ | Ideal for small-scale experiments |
| minGRU-pytorch | github.com/lucidrains/minGRU-pytorch | Modern efficient RNN | code/minGRU-pytorch/ | Includes hybrid training |
| RENN | github.com/google-research/reverse-engineering-neural-networks | RNN attractor analysis toolkit | code/renn/ | **Critical** - fixed-point finding for RNNs |
| PyTorch word_language_model | github.com/pytorch/examples | RNN + Transformer in one codebase | code/pytorch-examples/ | **Critical** - unified training framework |

See `code/README.md` for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
1. Used paper-finder service (93 results, filtered by relevance)
2. Web search across arXiv, Semantic Scholar, ACL Anthology
3. Targeted searches for: attractor dynamics, concept attractors, persona collapse, mode collapse, RNN vs transformer comparison
4. GitHub search for relevant implementations

### Selection Criteria
- Direct relevance to architecture-specific attractor formation
- Papers comparing RNN vs Transformer behavior
- Papers formalizing attractor dynamics in neural networks
- Papers on mode/persona collapse as attractor phenomena
- Code repos enabling small-scale architecture comparison experiments

### Challenges Encountered
- HuggingFace Hub had intermittent 504 timeouts during dataset downloads
- WikiText-2 S3 endpoint had SSL certificate issues (used HF parquet API instead)
- No public code for Concept Attractors or Attractor Cycles papers
- PersonaChat dataset script deprecated on HuggingFace

### Gaps and Workarounds
- **Missing code for key papers**: Will need to reimplement attractor analysis from paper descriptions
- **No direct RNN attractor analysis tools for text**: Adapt Concept Attractors methodology to RNN hidden states
- **Limited persona/preference datasets**: Created synthetic preference questions dataset

## Recommendations for Experiment Design

### 1. Primary Dataset(s)
- **TinyStories** for training (simple, controlled, sufficient for small models)
- **WikiText-2** for validation/comparison
- **Synthetic preference questions** for attractor probing

### 2. Baseline Methods
- **nanoGPT** (~1-10M params transformer)
- **minGRU** (comparable RNN)
- **Hybrid** (minGRU + attention) as intermediate architecture

### 3. Evaluation Metrics
- Hidden state cosine similarity across layers (attractor convergence)
- 2-periodicity degree for iterative tasks
- Default response distribution entropy
- DTW for representation trajectory comparison

### 4. Code to Adapt/Reuse
- **PyTorch word_language_model**: Best starting point - unified RNN/Transformer training on WikiText-2
- **nanoGPT**: Training framework, model architecture, data pipeline
- **minGRU-pytorch**: RNN implementation with compatible training interface
- **RENN toolkit**: Fixed-point analysis and attractor characterization for RNNs
- **rnn-icrag**: RNN vs Transformer comparison infrastructure

### 5. Experiment Plan Outline
1. Train matched-parameter nanoGPT and minGRU on TinyStories
2. Extract hidden representations for concept-related prompts
3. Compute attractor metrics (convergence, clustering, periodicity)
4. Compare attractor structures between architectures
5. Test default responses to preference questions
6. Scale experiments (1M → 5M → 10M params) to test convergence hypothesis
