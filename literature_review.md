# Literature Review: Architecture-Specific Data Attractors

## Research Area Overview

This review examines whether different neural network architectures (RNNs vs. Transformers/LLMs) develop distinct attractors or default responses when exposed to similar data and training regimes. The research spans three interconnected areas: (1) attractor dynamics in neural networks, (2) architectural differences between RNNs and Transformers, and (3) mode/persona collapse phenomena in language models.

## Key Papers

### 1. Concept Attractors in LLMs and their Applications
- **Authors**: Chytas, Singh (UW-Madison)
- **Year**: 2025 | **Source**: arXiv 2601.11575
- **Key Contribution**: Formalizes the observation that LLM layers implement Iterated Function Systems (IFS), mapping semantically related prompts to concept-specific "attractor" regions in latent space.
- **Methodology**: 
  - Model each concept's transformation as an affine contractive map: φ_eff = M_eff·V + t_eff
  - Fit parameters by minimizing discrepancy between LLM's observed states and iterated map predictions
  - Attractor = fixed point V* to which all trajectories converge
- **Key Results**:
  - Different concept families form attractors at different layers (programming: layer 19, literature: layer 18, natural languages: layer 27 in Llama 3.1 8B)
  - Fractal-like hierarchical structure within attractors
  - Training-free steering by adding/subtracting attractor vectors matches trainable methods
  - Phenomenon observed across Llama 3.1, Gemma, and Qwen families
- **Datasets Used**: TOFU benchmark, custom prompts for fictional universes, programming tasks
- **Code Available**: No public repository found
- **Relevance**: Establishes the theoretical framework for understanding attractors in LLMs. Critical gap: only studies transformer architectures. Our research extends this to RNNs.

### 2. Unveiling Attractor Cycles in LLMs: Successive Paraphrasing
- **Authors**: Wang, Li, Yan, Cheng, Zhang
- **Year**: 2025 | **Source**: ACL 2025, arXiv 2502.15208
- **Key Contribution**: Demonstrates that LLMs converge to stable 2-period attractor cycles when performing successive paraphrasing.
- **Methodology**:
  - Frame successive paraphrasing as discrete dynamical system: T_{n+1} = P(T_n)
  - Measure textual distance using normalized Levenshtein edit distance
  - Define "2-periodicity degree" metric: τ = 1 - (1/(M-2)) Σ d(T_i, T_{i-2})
  - Test across 8 LLMs, 15 rounds of paraphrasing
- **Key Results**:
  - Consistent 2-period cycles across all tested models (Mistral, Llama, Qwen, GPT-4o)
  - Pattern persists across languages (English, Chinese), text lengths, and temperatures
  - LLMs grow increasingly confident in narrow solution set
  - Extends beyond paraphrasing to any invertible task
- **Datasets Used**: MAGE dataset (1000 sentences, 300 paragraphs), WMT 2019
- **Code Available**: Promised but not yet released
- **Relevance**: Demonstrates attractor behavior in the output space of LLMs. Key gap: only tests transformer-based LLMs. Do RNNs exhibit similar or different attractor cycles?

### 3. Comparing Generalization: Transformer vs. RNN in Attractor Dynamics
- **Authors**: Fukushima, Tani (OIST)
- **Year**: 2023 | **Source**: arXiv 2311.10763
- **Key Contribution**: **Most directly relevant paper.** Directly compares how RNNs and Transformers learn attractor dynamics from limited data.
- **Methodology**:
  - Train both architectures on point attractors and cyclic attractors (2D dynamical systems)
  - Autoregressive prediction: given initial position, predict trajectory for T timesteps
  - Evaluate using Dynamic Time Warping (DTW)
  - Systematically vary number of training sequences (1-50)
- **Key Results**:
  - **RNN converges to correct attractor dynamics with only 2-4 training sequences**
  - **Transformer requires 25-50 sequences for comparable performance**
  - For cyclic attractors, RNN produces limit cycles even from single training sequence
  - Transformer fails to reproduce attractors even with 9 sequences
  - Results persist across dropout rate variations
  - Attributed to RNN's "recurrent inductive bias" via shared weights
- **Datasets Used**: Synthetic 2D dynamical systems (point and cyclic attractors)
- **Code Available**: Not found
- **Relevance**: Directly demonstrates architecture-specific attractor learning. Supports hypothesis that RNNs and Transformers develop different attractor behaviors.

### 4. RNNs Are Not Transformers (Yet): In-Context Retrieval
- **Authors**: Wen, Dang, Lyu (Tsinghua/Princeton)
- **Year**: 2024 | **Source**: ICLR 2025, arXiv 2402.18510
- **Key Contribution**: Proves fundamental representation power gap between RNNs and Transformers.
- **Methodology**:
  - Formal complexity-theoretic analysis of RNN vs Transformer expressiveness
  - Experiments on IsTree (graph algorithm) and HotPot-QA
  - Tests Mamba (RNN) vs LLaMA 2 (Transformer) at 0.5M-2M parameters
- **Key Results**:
  - RNN + CoT has O(log n) bit memory, exponentially weaker than Transformer + CoT
  - Key bottleneck: RNNs cannot perform in-context retrieval due to constant-memory constraint
  - Adding single Transformer layer to RNN closes representation gap
  - Results apply to Mamba, RWKV, Linear Transformers
- **Datasets Used**: Synthetic IsTree graphs, HotPot-QA
- **Code Available**: Yes - https://github.com/dangxingyu/rnn-icrag
- **Relevance**: Proves architectures have fundamentally different capabilities, which should lead to different attractor formation. The in-context retrieval bottleneck in RNNs may cause different default behaviors.

### 5. Transformers are Multi-State RNNs
- **Authors**: Various
- **Year**: 2024 | **Source**: arXiv 2401.06104
- **Key Contribution**: Shows that Transformer attention can be viewed as a multi-state RNN, bridging the two architectures conceptually.
- **Relevance**: Provides theoretical framework for understanding when/how the two architectures might converge in behavior.

### 6. Sycophancy Origins in LLMs
- **Authors**: Various
- **Year**: 2025 | **Source**: arXiv 2508.02087
- **Key Contribution**: Shows LLMs develop systematic sycophantic behavior (agreeing with incorrect user opinions). User opinions actively suppress model's learned knowledge in later layers.
- **Relevance**: Sycophancy can be viewed as a type of attractor - a default behavioral pattern. Architecture may influence susceptibility.

### 7. Mode Collapse / Persona Collapse
- **Sources**: arXiv 2510.01171 (Verbalized Sampling), HuggingFace blog (Persona Collapse Taxonomy)
- **Key Findings**:
  - Mode collapse in LLMs: RLHF alignment constrains models from generalizing over authorship
  - Persona collapse: models fail to maintain distinct personas under recursive pressure
  - Different LLM families show different collapse patterns
- **Relevance**: Mode/persona collapse are manifestations of attractor dynamics. Architecture may determine which attractor patterns emerge.

### 8. Interpreting RNN Behaviour via Excitable Network Attractors
- **Authors**: Various
- **Year**: 2018 | **Source**: arXiv 1807.10478
- **Key Contribution**: Methods for extracting attractor dynamics from trained RNNs using invariant sets in phase space.
- **Relevance**: Provides tools for analyzing RNN-specific attractor structures that can be compared with transformer attractors.

## Common Methodologies

- **Representation analysis**: Extract hidden states at each layer, measure convergence via cosine similarity or distance metrics (used in Concept Attractors, Attractor Cycles)
- **Dynamical systems framing**: Model network as iterative function system, identify fixed points and limit cycles (used in Concept Attractors, Attractor Cycles, RNN Attractors)
- **Few-shot generalization**: Compare architectures on ability to learn from limited examples (Fukushima & Tani)
- **Formal expressiveness analysis**: Prove representation power bounds using complexity theory (RNNs Are Not Transformers)

## Standard Baselines

- **Transformer**: GPT-2/nanoGPT for small-scale, Llama 3.1 8B for larger-scale
- **RNN**: Vanilla RNN, GRU, LSTM for traditional; Mamba, RWKV for modern
- **Hybrid**: RNN + single attention layer (shown to close representation gap)
- **Metrics**: DTW for trajectory comparison, Levenshtein distance for text, cosine similarity for representations

## Evaluation Metrics

- **Attractor convergence**: Cosine similarity between representations at each layer
- **2-periodicity degree**: Measures cyclic attractor patterns in iterative tasks
- **Dynamic Time Warping (DTW)**: Measures trajectory similarity for dynamical systems
- **Generalization in Learning (GIL)**: Performance vs. number of training examples

## Gaps and Opportunities

1. **No direct comparison of attractor formation in RNNs vs. Transformers for language tasks** - Fukushima & Tani compare on 2D dynamics, not text. Concept Attractors and Attractor Cycles only study transformers.
2. **No study of whether attractor cycles differ by architecture** - Do RNNs show 2-period cycles like transformers, or different period/structure?
3. **No study of whether "sufficiently capable" RNNs converge to same attractors as LLMs** - The research hypothesis directly addresses this.
4. **Limited understanding of how architectural inductive biases shape default responses** - Sycophancy and persona collapse studied per-model but not per-architecture-family.
5. **No comparison of attractor layer locations across architectures** - In transformers, different concepts form attractors at different layers. What about RNNs?

## Recommendations for Our Experiment

### Recommended Datasets
1. **TinyStories** (primary) - Simple enough to train small models from scratch, rich enough to form meaningful attractors
2. **WikiText-2** (secondary) - Standard benchmark for language modeling comparison
3. **Synthetic preference questions** - Direct probe of default/attractor responses
4. **MAGE dataset** - For replicating attractor cycles experiments

### Recommended Baselines
1. **nanoGPT** (small transformer, ~1-10M params)
2. **minGRU** (modern efficient RNN, comparable size)
3. **Hybrid** (RNN + attention layer) as intermediate point

### Recommended Metrics
1. Cosine similarity of hidden representations (concept attractor analysis)
2. 2-periodicity degree (attractor cycle analysis)
3. Default response entropy (do architectures converge to narrower defaults?)
4. DTW for trajectory comparison in representation space

### Methodological Considerations
- Match parameter counts between RNN and Transformer models for fair comparison
- Train on identical data with identical hyperparameters where possible
- Extract representations at multiple layers/timesteps for attractor analysis
- Use both trained and untrained models to separate architectural priors from learned behavior
- Test at multiple scales (1M, 5M, 10M params) to assess scaling effects on attractor formation
