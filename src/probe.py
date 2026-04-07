"""
Experiment 3: Probe trained models for default/preference responses.
Generate completions from preference-style prompts and analyze convergence.
"""

import os
import sys
import json
import random
import numpy as np
import torch
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from models import build_models
from data import SimpleTokenizer


# Preference-style story prompts that invite default responses
STORY_PROMPTS = [
    # Food/dessert related
    "once upon a time there was a girl who loved to eat",
    "the best food in the world is",
    "for dessert they had",
    "the cake was made of",
    "she wanted to eat something sweet like",

    # Color related
    "the sky was a beautiful shade of",
    "her favorite color was",
    "the dress was a pretty",
    "he painted the wall a nice",

    # Animal related
    "in the forest there was a friendly",
    "the little boy had a pet",
    "the animal she loved most was a",

    # Season/nature
    "the best time of year was",
    "in the spring the flowers were",
    "on a sunny day they went to the",

    # General preferences
    "the most important thing in life is",
    "the best way to be happy is to",
    "what she wanted most was",
    "the most beautiful thing was",
    "he always dreamed of",
]


def generate_text(model, tokenizer, prompt, device, max_new_tokens=30,
                  temperature=0.8, top_k=40):
    """Generate text continuation from a prompt."""
    model.eval()
    # Encode without <eos> so the model continues the text
    words = prompt.lower().split()
    ids = [tokenizer.bos_id]
    for w in words:
        ids.append(tokenizer.word2idx.get(w, tokenizer.unk_id))
    # Don't add <eos> - we want the model to continue

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    prompt_len = len(ids)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if input_ids.shape[1] > 200:
                input_ids = input_ids[:, -200:]

            logits = model(input_ids)
            logits = logits[:, -1, :] / temperature

            # Suppress <bos> and <pad> tokens
            logits[:, tokenizer.bos_id] = float('-inf')
            logits[:, tokenizer.pad_id] = float('-inf')

            # Top-k sampling
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, next_id], dim=1)

            if next_id.item() == tokenizer.eos_id:
                break

    generated_ids = input_ids[0].tolist()
    # Decode only the generated part (after prompt)
    generated_words = []
    for idx in generated_ids[prompt_len:]:
        if idx == tokenizer.eos_id:
            break
        if idx in (tokenizer.pad_id, tokenizer.bos_id):
            continue
        generated_words.append(tokenizer.idx2word.get(idx, "<unk>"))

    full_text = prompt + " " + " ".join(generated_words)
    return full_text.strip()


def probe_models(model_configs, tokenizer, device, num_samples=20):
    """Probe all models with preference prompts and collect responses."""
    all_results = {}

    for model_name, model in model_configs.items():
        print(f"\n{'='*50}")
        print(f"Probing: {model_name}")
        print(f"{'='*50}")

        model = model.to(device)
        model.eval()

        model_results = {}
        for prompt in STORY_PROMPTS:
            responses = []
            for i in range(num_samples):
                text = generate_text(model, tokenizer, prompt, device,
                                     max_new_tokens=30, temperature=0.8)
                responses.append(text)

            model_results[prompt] = responses
            # Show first 3
            print(f"\n  Prompt: '{prompt}'")
            for r in responses[:3]:
                print(f"    -> {r[:100]}")

        all_results[model_name] = model_results

    return all_results


def analyze_word_attractors(all_results):
    """Analyze which words each architecture tends to generate most."""
    print("\n" + "=" * 60)
    print("WORD ATTRACTOR ANALYSIS")
    print("=" * 60)

    analysis = {}

    for model_name, prompts_dict in all_results.items():
        print(f"\n--- {model_name} ---")
        word_counts = Counter()
        # Count all generated words (excluding prompt words)
        for prompt, responses in prompts_dict.items():
            prompt_words = set(prompt.lower().split())
            for resp in responses:
                words = resp.lower().split()
                for w in words:
                    if w not in prompt_words and len(w) > 2:
                        word_counts[w] += 1

        top_words = word_counts.most_common(30)
        print(f"  Top generated words: {top_words[:15]}")
        analysis[model_name] = {
            "top_words": top_words,
            "total_unique_words": len(word_counts),
            "total_words": sum(word_counts.values()),
        }

    return analysis


def compute_response_entropy(all_results):
    """Compute entropy of next-word distributions per prompt per model."""
    print("\n" + "=" * 60)
    print("RESPONSE ENTROPY ANALYSIS")
    print("=" * 60)

    entropy_results = {}

    for model_name, prompts_dict in all_results.items():
        entropies = []
        for prompt, responses in prompts_dict.items():
            # Get first content word after prompt in each response
            prompt_len = len(prompt.lower().split())
            first_words = []
            for resp in responses:
                words = resp.lower().split()
                if len(words) > prompt_len + 1:
                    # Skip <bos> token
                    content_words = [w for w in words[prompt_len:] if w not in
                                     ("<bos>", "<eos>", "<pad>", "<unk>")]
                    if content_words:
                        first_words.append(content_words[0])

            if first_words:
                counts = Counter(first_words)
                total = sum(counts.values())
                probs = [c / total for c in counts.values()]
                entropy = -sum(p * np.log2(p) for p in probs if p > 0)
                entropies.append(entropy)

        avg_entropy = np.mean(entropies) if entropies else 0
        entropy_results[model_name] = {
            "mean_entropy": float(avg_entropy),
            "per_prompt_entropies": [float(e) for e in entropies],
        }
        print(f"  {model_name}: mean response entropy = {avg_entropy:.3f} bits")

    return entropy_results


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer = SimpleTokenizer.load("results/tokenizer.json")

    # Load trained models
    model_configs = {}
    for scale in ["small", "medium"]:
        models = build_models(tokenizer.vocab_size, scale=scale)
        for name, model in models.items():
            full_name = name if scale == "small" else f"{name}_medium"
            ckpt = f"results/checkpoints/{full_name}_best.pt"
            if os.path.exists(ckpt):
                model.load_state_dict(torch.load(ckpt, weights_only=True,
                                                  map_location=device))
                model_configs[full_name] = model
                print(f"Loaded {full_name} from {ckpt}")
            else:
                print(f"WARNING: {ckpt} not found, skipping")

    if not model_configs:
        print("ERROR: No trained models found. Run train.py first.")
        return

    # Probe models
    probe_results = probe_models(model_configs, tokenizer, device, num_samples=20)

    # Save raw results
    with open("results/exp3_probe_results.json", "w") as f:
        json.dump(probe_results, f, indent=2)

    # Analysis
    word_analysis = analyze_word_attractors(probe_results)
    entropy_analysis = compute_response_entropy(probe_results)

    # Save analysis
    with open("results/exp3_analysis.json", "w") as f:
        json.dump({
            "word_attractors": {k: {"top_words": v["top_words"][:30],
                                     "total_unique": v["total_unique_words"],
                                     "total": v["total_words"]}
                                 for k, v in word_analysis.items()},
            "entropy": entropy_analysis,
        }, f, indent=2)

    print("\nResults saved to results/exp3_*.json")


if __name__ == "__main__":
    main()
