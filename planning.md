# Research Plan: Architecture-Specific Data Attractors

## Motivation & Novelty Assessment

### Why This Research Matters
Frontier LLMs exhibit a striking convergence in "persona" responses — nearly all claim tiramisu as their favorite dessert, prefer blue, enjoy autumn, etc. This "persona collapse" suggests transformer architectures develop shared attractors in response space when trained on similar internet-scale data. Understanding whether this is an artifact of the transformer architecture, the training data, or the RLHF alignment process has implications for AI safety (monoculture risks), model diversity, and our understanding of how neural network architecture shapes emergent behavior.

### Gap in Existing Work
- Chytas & Singh (2025) formalized concept attractors in LLMs but only studied transformers
- Fukushima & Tani (2023) compared RNN vs Transformer attractor dynamics but only on 2D dynamical systems, not language
- Wang et al. (2025) showed attractor cycles in paraphrasing but only for transformer LLMs
- **No study has directly compared attractor/default response formation between RNNs and Transformers in the language domain**
- **No study has tested whether "sufficiently capable" RNNs converge to the same persona attractors as transformers**

### Our Novel Contribution
We directly test whether architectural differences (RNN vs Transformer) produce different default/"persona" responses when trained on identical data. We also probe real frontier LLMs to establish the baseline attractor pattern, and test whether scaling RNNs toward transformer-level capability shifts their attractors toward the transformer attractor basin.

### Experiment Justification

- **Experiment 1 (Real LLM Survey)**: Establishes the ground truth that frontier transformer LLMs converge on similar default responses. Uses OpenRouter API to query multiple models with preference questions. This is the baseline phenomenon we're studying.

- **Experiment 2 (Train Matched Models)**: Tests whether architecture alone (controlling for data and training) produces different attractors. We train size-matched RNN (GRU) and Transformer models on identical TinyStories data.

- **Experiment 3 (Default Response Probing)**: After training, we prompt both architectures with preference/opinion questions and measure whether their completions diverge systematically.

- **Experiment 4 (Representation Attractor Analysis)**: We extract hidden representations from both architectures for identical prompts and measure attractor convergence using cosine similarity clustering, to understand the mechanistic basis of any response differences.

## Research Question
Do different neural network architectures (RNNs vs Transformers) develop distinct default response attractors when trained on similar data? Would a sufficiently capable RNN converge to the same attractors as transformer LLMs?

## Hypothesis Decomposition
1. **H1**: Frontier transformer LLMs show strong convergence on default preference responses (the "tiramisu effect")
2. **H2**: Small RNN and Transformer models trained on identical data develop different default completions for preference-style prompts
3. **H3**: The representation space attractors (hidden state clustering patterns) differ structurally between RNN and Transformer architectures
4. **H4**: Scaling RNN models toward higher capability shifts their attractor patterns toward (but not necessarily to) the transformer attractor basin

## Proposed Methodology

### Approach
Multi-pronged empirical study combining:
1. API-based survey of real frontier LLMs (establishing the phenomenon)
2. Controlled training experiments with matched architectures (isolating architecture effects)
3. Representation-level analysis (mechanistic understanding)

### Experimental Steps

**Exp 1: Frontier LLM Attractor Survey**
- Query 6-8 frontier LLMs via OpenRouter with 10 preference questions (including "favorite dessert")
- Run each query 5 times per model (temperature=0.7) to measure response distribution
- Quantify convergence: what fraction of models give the same top answer?

**Exp 2: Matched Architecture Training**
- Train a small Transformer (~5M params) and GRU-RNN (~5M params) on TinyStories
- Match: same tokenizer, same data, same training steps, same optimizer
- Also train at ~15M params to test scale effects

**Exp 3: Default Response Probing**
- Prompt trained models with preference questions as story starters
- Generate 50 completions per question per model (varied sampling)
- Analyze response distributions, measure entropy of response categories

**Exp 4: Representation Analysis**
- Feed identical prompts through both architectures
- Extract hidden states at each layer/timestep
- Measure: cosine similarity convergence, clustering coefficient, attractor basin size

### Baselines
- Random baseline (uniform distribution over response categories)
- Single-architecture baseline (variation within architecture across seeds)
- Cross-architecture comparison (our main analysis)

### Evaluation Metrics
- **Response convergence**: fraction of models giving same top-1 answer
- **Response entropy**: Shannon entropy of answer distribution per question
- **Cosine similarity**: between hidden representations for related prompts
- **Jensen-Shannon divergence**: between response distributions of different architectures
- **Effect size**: Cohen's d for cross-architecture differences

### Statistical Analysis Plan
- Chi-squared tests for categorical response distributions
- Mann-Whitney U tests for continuous metrics
- Bootstrap confidence intervals (1000 resamples)
- Bonferroni correction for multiple comparisons
- Significance level: α = 0.05

## Expected Outcomes
- **If H1 confirmed**: Frontier LLMs show <2 bits entropy on preference questions (strong convergence)
- **If H2 confirmed**: RNN and Transformer completions have JSD > 0.1 (meaningfully different distributions)
- **If H3 confirmed**: Hidden state clustering patterns differ qualitatively between architectures
- **If H4 confirmed**: Larger RNNs show lower JSD from transformers than smaller RNNs

## Timeline and Milestones
1. Environment setup + Exp 1 (LLM survey): 30 min
2. Data prep + model training (Exp 2): 60 min
3. Response probing (Exp 3): 30 min
4. Representation analysis (Exp 4): 30 min
5. Statistical analysis + visualization: 30 min
6. Documentation: 30 min

## Potential Challenges
- Small models may not develop meaningful "preferences" — mitigate by using story-completion prompts rather than direct questions
- TinyStories may not contain enough preference-laden content — supplement with WikiText-2
- RNN training may be unstable at larger scales — use gradient clipping, careful LR scheduling
- API rate limits — cache all responses, use batch queries

## Success Criteria
1. Clear evidence that frontier LLMs converge on similar responses (establishing the phenomenon)
2. Statistically significant difference in default responses between RNN and Transformer trained on same data
3. Mechanistic explanation via representation analysis
4. Preliminary evidence on scale-dependent convergence
