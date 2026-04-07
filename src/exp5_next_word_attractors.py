"""
Experiment 5: Direct next-word attractor comparison.
For each prompt, compare the top-k next-word probability distributions
between architectures using greedy decoding (temperature=0).
This reveals the "default attractors" of each architecture.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from models import build_models
from data import SimpleTokenizer


ATTRACTOR_PROMPTS = [
    # Food
    "once upon a time there was a girl who loved to eat",
    "for dessert they had",
    "the cake was made of",
    # Colors
    "her favorite color was",
    "the sky was a beautiful shade of",
    # Animals
    "in the forest there was a friendly",
    "the little boy had a pet",
    # General
    "the most important thing in life is",
    "she was very",
    "he wanted to",
    "they decided to",
    "the little girl named",
]


def get_top_k_predictions(model, tokenizer, prompt, device, k=10):
    """Get top-k next word predictions with probabilities."""
    model.eval()
    words = prompt.lower().split()
    ids = [tokenizer.bos_id]
    for w in words:
        ids.append(tokenizer.word2idx.get(w, tokenizer.unk_id))

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(input_ids)
        probs = F.softmax(logits[0, -1, :], dim=-1)

        # Get top-k
        top_probs, top_ids = torch.topk(probs, k)
        predictions = []
        for prob, idx in zip(top_probs, top_ids):
            word = tokenizer.idx2word.get(idx.item(), "<unk>")
            predictions.append((word, float(prob)))

    return predictions


def compute_jsd(p, q):
    """Jensen-Shannon divergence between two probability dicts."""
    all_words = set(p.keys()) | set(q.keys())
    p_arr = np.array([p.get(w, 1e-10) for w in all_words])
    q_arr = np.array([q.get(w, 1e-10) for w in all_words])

    # Normalize
    p_arr = p_arr / p_arr.sum()
    q_arr = q_arr / q_arr.sum()

    m = 0.5 * (p_arr + q_arr)
    jsd = 0.5 * np.sum(p_arr * np.log2(p_arr / m + 1e-10)) + \
          0.5 * np.sum(q_arr * np.log2(q_arr / m + 1e-10))
    return float(jsd)


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = SimpleTokenizer.load("results/tokenizer.json")

    # Load small-scale models only (best trained)
    models_dict = build_models(tokenizer.vocab_size, scale="small")
    model_configs = {}
    for name, model in models_dict.items():
        ckpt = f"results/checkpoints/{name}_best.pt"
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, weights_only=True, map_location=device))
            model.to(device)
            model_configs[name] = model
            print(f"Loaded {name}")

    # Get predictions
    all_predictions = {}
    for model_name, model in model_configs.items():
        all_predictions[model_name] = {}
        for prompt in ATTRACTOR_PROMPTS:
            preds = get_top_k_predictions(model, tokenizer, prompt, device, k=10)
            all_predictions[model_name][prompt] = preds

    # Display results
    print("\n" + "=" * 70)
    print("NEXT-WORD ATTRACTOR ANALYSIS")
    print("=" * 70)

    for prompt in ATTRACTOR_PROMPTS:
        print(f"\nPrompt: '{prompt} ___'")
        print("-" * 60)
        for model_name in model_configs:
            preds = all_predictions[model_name][prompt]
            top3 = ", ".join([f"{w}({p:.2%})" for w, p in preds[:5]])
            print(f"  {model_name:15s}: {top3}")

    # Compute pairwise JSD
    print("\n" + "=" * 70)
    print("JENSEN-SHANNON DIVERGENCE BETWEEN ARCHITECTURES")
    print("=" * 70)

    arch_names = list(model_configs.keys())
    jsd_matrix = defaultdict(dict)
    per_prompt_jsd = defaultdict(lambda: defaultdict(dict))

    for i, a1 in enumerate(arch_names):
        for j, a2 in enumerate(arch_names):
            if i < j:
                jsds = []
                for prompt in ATTRACTOR_PROMPTS:
                    p1 = dict(all_predictions[a1][prompt])
                    p2 = dict(all_predictions[a2][prompt])
                    jsd = compute_jsd(p1, p2)
                    jsds.append(jsd)
                    per_prompt_jsd[prompt][f"{a1}-{a2}"] = jsd

                mean_jsd = np.mean(jsds)
                jsd_matrix[a1][a2] = mean_jsd
                jsd_matrix[a2][a1] = mean_jsd
                print(f"  {a1} vs {a2}: JSD = {mean_jsd:.4f} (±{np.std(jsds):.4f})")

    # Check which prompts show most/least divergence
    print("\n" + "=" * 70)
    print("PROMPTS RANKED BY CROSS-ARCHITECTURE DIVERGENCE")
    print("=" * 70)

    prompt_divergence = {}
    for prompt in ATTRACTOR_PROMPTS:
        mean_jsd = np.mean(list(per_prompt_jsd[prompt].values()))
        prompt_divergence[prompt] = mean_jsd

    for prompt, div in sorted(prompt_divergence.items(), key=lambda x: -x[1]):
        print(f"  JSD={div:.4f}: '{prompt}'")

    # Check if top-1 predictions agree across architectures
    print("\n" + "=" * 70)
    print("TOP-1 PREDICTION AGREEMENT")
    print("=" * 70)

    agreement_count = 0
    total = 0
    for prompt in ATTRACTOR_PROMPTS:
        top1s = {}
        for model_name in model_configs:
            top1s[model_name] = all_predictions[model_name][prompt][0][0]

        all_same = len(set(top1s.values())) == 1
        if all_same:
            agreement_count += 1
        total += 1

        marker = "AGREE" if all_same else "DIFFER"
        top1_str = ", ".join(f"{m}={w}" for m, w in top1s.items())
        print(f"  [{marker}] '{prompt}' -> {top1_str}")

    print(f"\n  Agreement rate: {agreement_count}/{total} ({agreement_count/total:.0%})")

    # Save results
    results = {
        "predictions": {m: {p: [(w, prob) for w, prob in preds]
                              for p, preds in prompts.items()}
                         for m, prompts in all_predictions.items()},
        "jsd_pairwise": {f"{a1}-{a2}": float(v)
                          for a1 in jsd_matrix for a2, v in jsd_matrix[a1].items()
                          if a1 < a2},
        "per_prompt_divergence": {p: float(v) for p, v in prompt_divergence.items()},
        "top1_agreement_rate": agreement_count / total,
    }

    with open("results/exp5_next_word_attractors.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to results/exp5_next_word_attractors.json")


if __name__ == "__main__":
    main()
