# Architecture-Specific Data Attractors

**Research Question**: Do different neural network architectures develop distinct default responses ("attractors") when trained on identical data?

## Key Findings

- **Frontier LLMs show strong persona convergence**: GPT-4o, GPT-4o-mini, and GPT-4.1-mini unanimously converge on the same answers for 7/10 preference questions (chocolate lava cake, blue, dolphins, early morning, 7, etc.)
- **Architecture determines attractors**: Transformer, GRU, and LSTM models trained on identical TinyStories data agree on their top-1 prediction only **25% of the time** (JSD = 0.34 between Transformer and GRU)
- **Transformers have sharper attractors**: Transformers concentrate 43% probability on a single prediction where GRU gives 27% — stronger persona collapse tendency
- **Transformers collapse representation space**: Within-concept similarity 0.94, between-concept 0.95 (negative concept separation). LSTMs maintain better concept separation (+0.06) despite lower capability
- **Matching capability ≠ matching attractors**: GRU matches Transformer perplexity (28.6 vs 28.3) but diverges substantially in default responses

## Implications

A "sufficiently good RNN" would likely *not* converge to the same attractors as transformer LLMs. Architectural inductive biases shape which defaults emerge independently of capability. The transformer's attention mechanism creates a "universal attractor basin" that facilitates persona collapse.

## Reproducing

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install torch numpy matplotlib scipy scikit-learn pandas requests tiktoken pyarrow openai

# Run experiments
python src/train.py           # Train matched models (~10 min with GPU)
python src/probe.py           # Probe for default responses
python src/exp4_representation_analysis.py  # Representation analysis
python src/exp5_next_word_attractors.py     # Next-word attractor comparison
python src/exp1_llm_survey.py              # Survey frontier LLMs (needs API key)
python src/visualize.py       # Generate figures
```

## File Structure

```
├── REPORT.md                    # Full research report
├── planning.md                  # Research plan
├── src/
│   ├── models.py                # Transformer, GRU, LSTM model definitions
│   ├── data.py                  # Data loading and tokenization
│   ├── train.py                 # Training script
│   ├── probe.py                 # Text generation probing
│   ├── exp1_llm_survey.py       # Frontier LLM survey
│   ├── exp4_representation_analysis.py  # Hidden state analysis
│   ├── exp5_next_word_attractors.py     # Next-word distribution comparison
│   └── visualize.py             # Visualization and statistics
├── results/                     # Experimental results (JSON)
├── figures/                     # Generated plots
├── datasets/                    # TinyStories, WikiText-2, synthetic prompts
├── papers/                      # 13 research papers
└── code/                        # Reference implementations
```

See [REPORT.md](REPORT.md) for full methodology, results, and discussion.
