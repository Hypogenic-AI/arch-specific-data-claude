"""
Experiment 4: Representation-level attractor analysis.
Extract hidden states from matched models and compare attractor structure.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from models import build_models, TransformerLM, GRULM, LSTMLM
from data import SimpleTokenizer


# Concept-related prompts grouped by topic
CONCEPT_PROMPTS = {
    "food": [
        "the cake was very",
        "she ate some delicious",
        "the food was really",
        "they cooked a nice",
        "for breakfast they had",
    ],
    "animals": [
        "the little dog was",
        "a cute cat sat",
        "the bird flew over",
        "the fish swam in",
        "the rabbit hopped around",
    ],
    "colors": [
        "the sky was bright",
        "her dress was a",
        "the flower was a",
        "the ball was red",
        "the grass was very",
    ],
    "emotions": [
        "she was very happy",
        "he felt so sad",
        "they were really excited",
        "the girl was scared",
        "everyone was very proud",
    ],
    "actions": [
        "the boy started to",
        "she began to run",
        "they decided to play",
        "he wanted to help",
        "she tried to find",
    ],
}


def extract_hidden_states(model, tokenizer, prompts, device):
    """Extract final hidden states for a list of prompts."""
    model.eval()
    model.to(device)
    hidden_states = []

    with torch.no_grad():
        for prompt in prompts:
            ids = tokenizer.encode(prompt)
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            _, h = model(input_ids, return_hidden=True)
            # Take the last token's hidden state
            last_hidden = h[0, -1, :].cpu().numpy()
            hidden_states.append(last_hidden)

    return np.array(hidden_states)


def compute_attractor_metrics(hidden_states_by_concept):
    """Compute attractor-related metrics from hidden states."""
    metrics = {}

    all_states = []
    concept_labels = []
    for concept, states in hidden_states_by_concept.items():
        all_states.append(states)
        concept_labels.extend([concept] * len(states))
    all_states = np.concatenate(all_states, axis=0)

    # 1. Within-concept cosine similarity (attractor tightness)
    within_sims = {}
    for concept, states in hidden_states_by_concept.items():
        norms = np.linalg.norm(states, axis=1, keepdims=True) + 1e-8
        normed = states / norms
        sim_matrix = normed @ normed.T
        # Average off-diagonal
        n = len(states)
        if n > 1:
            mask = ~np.eye(n, dtype=bool)
            avg_sim = sim_matrix[mask].mean()
        else:
            avg_sim = 1.0
        within_sims[concept] = float(avg_sim)

    metrics["within_concept_similarity"] = within_sims
    metrics["mean_within_similarity"] = float(np.mean(list(within_sims.values())))

    # 2. Between-concept cosine similarity
    concept_centroids = {}
    for concept, states in hidden_states_by_concept.items():
        centroid = states.mean(axis=0)
        concept_centroids[concept] = centroid

    concepts = list(concept_centroids.keys())
    between_sims = {}
    for i, c1 in enumerate(concepts):
        for j, c2 in enumerate(concepts):
            if i < j:
                v1 = concept_centroids[c1]
                v2 = concept_centroids[c2]
                sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))
                between_sims[f"{c1}-{c2}"] = sim

    metrics["between_concept_similarity"] = between_sims
    metrics["mean_between_similarity"] = float(np.mean(list(between_sims.values())))

    # 3. Silhouette-like score (within vs between separation)
    separation = metrics["mean_within_similarity"] - metrics["mean_between_similarity"]
    metrics["concept_separation"] = float(separation)

    # 4. Variance within concepts (attractor basin size)
    variances = {}
    for concept, states in hidden_states_by_concept.items():
        centroid = states.mean(axis=0)
        dists = np.linalg.norm(states - centroid, axis=1)
        variances[concept] = float(np.mean(dists))
    metrics["within_concept_variance"] = variances
    metrics["mean_within_variance"] = float(np.mean(list(variances.values())))

    return metrics


def run_analysis(model_configs, tokenizer, device):
    """Run full representation analysis across all models."""
    all_metrics = {}

    for model_name, model in model_configs.items():
        print(f"\nAnalyzing: {model_name}")

        hidden_by_concept = {}
        for concept, prompts in CONCEPT_PROMPTS.items():
            states = extract_hidden_states(model, tokenizer, prompts, device)
            hidden_by_concept[concept] = states
            print(f"  {concept}: states shape = {states.shape}")

        metrics = compute_attractor_metrics(hidden_by_concept)
        all_metrics[model_name] = metrics

        print(f"  Mean within-concept similarity: {metrics['mean_within_similarity']:.4f}")
        print(f"  Mean between-concept similarity: {metrics['mean_between_similarity']:.4f}")
        print(f"  Concept separation: {metrics['concept_separation']:.4f}")
        print(f"  Mean within-concept variance: {metrics['mean_within_variance']:.4f}")

    return all_metrics


def compare_architectures(all_metrics):
    """Compare attractor properties across architecture types."""
    print("\n" + "=" * 60)
    print("ARCHITECTURE COMPARISON")
    print("=" * 60)

    comparison = {}

    # Group by architecture type
    arch_groups = defaultdict(list)
    for model_name, metrics in all_metrics.items():
        if "transformer" in model_name:
            arch_groups["transformer"].append((model_name, metrics))
        elif "gru" in model_name:
            arch_groups["gru"].append((model_name, metrics))
        elif "lstm" in model_name:
            arch_groups["lstm"].append((model_name, metrics))

    for arch, entries in arch_groups.items():
        within_sims = [m["mean_within_similarity"] for _, m in entries]
        between_sims = [m["mean_between_similarity"] for _, m in entries]
        separations = [m["concept_separation"] for _, m in entries]
        variances = [m["mean_within_variance"] for _, m in entries]

        comparison[arch] = {
            "mean_within_similarity": float(np.mean(within_sims)),
            "mean_between_similarity": float(np.mean(between_sims)),
            "mean_separation": float(np.mean(separations)),
            "mean_variance": float(np.mean(variances)),
            "num_models": len(entries),
        }

        print(f"\n  {arch.upper()}:")
        print(f"    Within-concept sim: {np.mean(within_sims):.4f}")
        print(f"    Between-concept sim: {np.mean(between_sims):.4f}")
        print(f"    Concept separation: {np.mean(separations):.4f}")
        print(f"    Within-concept variance: {np.mean(variances):.4f}")

    return comparison


def main():
    torch.manual_seed(42)
    np.random.seed(42)

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
                print(f"Loaded {full_name}")

    if not model_configs:
        print("ERROR: No trained models found.")
        return

    # Run analysis
    all_metrics = run_analysis(model_configs, tokenizer, device)

    # Compare architectures
    comparison = compare_architectures(all_metrics)

    # Save results
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results = {
        "per_model_metrics": {k: {kk: convert(vv) for kk, vv in v.items()}
                               for k, v in all_metrics.items()},
        "architecture_comparison": comparison,
    }

    with open("results/exp4_representation_analysis.json", "w") as f:
        json.dump(results, f, indent=2, default=convert)

    print("\nResults saved to results/exp4_representation_analysis.json")


if __name__ == "__main__":
    main()
