"""
Visualization and statistical analysis for all experiments.
Generates publication-quality figures.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter


def plot_training_curves(results_path="results/training_results.json",
                          save_dir="figures"):
    """Plot training and validation loss curves for all models."""
    os.makedirs(save_dir, exist_ok=True)

    with open(results_path) as f:
        results = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Group by scale
    small_models = {k: v for k, v in results.items() if "medium" not in k}
    medium_models = {k: v for k, v in results.items() if "medium" in k}

    colors = {"transformer": "#2196F3", "gru": "#FF5722", "lstm": "#4CAF50",
              "transformer_medium": "#1565C0", "gru_medium": "#D84315",
              "lstm_medium": "#2E7D32"}

    for ax, (title, models) in zip(axes, [("Small Scale (~5M params)", small_models),
                                            ("Medium Scale (~15M params)", medium_models)]):
        for name, res in models.items():
            color = colors.get(name, "#000000")
            epochs = range(1, len(res["train_losses"]) + 1)
            ax.plot(epochs, res["train_losses"], '-', color=color, alpha=0.5,
                    label=f"{name} (train)")
            ax.plot(epochs, res["val_losses"], '--', color=color,
                    label=f"{name} (val)")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (Cross-Entropy)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved training_curves.png")


def plot_perplexity_comparison(results_path="results/training_results.json",
                                save_dir="figures"):
    """Bar chart comparing final perplexity across models."""
    os.makedirs(save_dir, exist_ok=True)

    with open(results_path) as f:
        results = json.load(f)

    names = list(results.keys())
    perplexities = [results[n]["best_perplexity"] for n in names]
    params = [results[n]["num_params"] for n in names]

    # Color by architecture
    colors = []
    for n in names:
        if "transformer" in n:
            colors.append("#2196F3")
        elif "gru" in n:
            colors.append("#FF5722")
        else:
            colors.append("#4CAF50")

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(names)), perplexities, color=colors)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f"{n}\n({p/1e6:.1f}M)" for n, p in zip(names, params)],
                       fontsize=9)
    ax.set_ylabel("Perplexity (lower is better)")
    ax.set_title("Model Perplexity Comparison: Architecture × Scale")
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, ppl in zip(bars, perplexities):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{ppl:.1f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "perplexity_comparison.png"), dpi=150,
                bbox_inches='tight')
    plt.close()
    print("Saved perplexity_comparison.png")


def plot_llm_survey_convergence(survey_path="results/exp1_llm_survey.json",
                                 save_dir="figures"):
    """Visualize convergence patterns from the LLM survey."""
    os.makedirs(save_dir, exist_ok=True)

    with open(survey_path) as f:
        results = json.load(f)

    # For each question, extract the key "attractor" word from responses
    questions = list(list(results.values())[0].keys())
    models = list(results.keys())

    # Manual extraction of dominant responses
    convergence_data = {}
    for q in questions:
        all_responses = []
        for model in models:
            for resp in results[model].get(q, []):
                all_responses.append(resp.lower())
        convergence_data[q] = all_responses

    # Create a heatmap-style visualization
    fig, ax = plt.subplots(figsize=(12, 6))

    # For each question, show whether models agree
    q_labels = [q.replace("What is your favorite ", "Fav ")
                 .replace("What color do you prefer?", "Color?")
                 .replace("Which season do you like best?", "Season?")
                 .replace("What is your favorite animal?", "Fav animal?")
                 .replace("What music genre do you prefer?", "Music?")
                 .replace("What time of day do you enjoy most?", "Time of day?")
                 .replace("Which planet is the most interesting?", "Planet?")
                 .replace("What is your favorite number?", "Fav number?")
                 .replace("What is the best programming language?", "Prog lang?")
                 .replace("What is the best way to learn?", "How to learn?")
                for q in questions]

    # Extract dominant answer per model per question
    dominant_answers = {}
    for q in questions:
        dominant_answers[q] = {}
        for model in models:
            responses = results[model].get(q, [])
            # Simple: just get the most common key words
            dominant_answers[q][model] = responses[0][:60] if responses else "N/A"

    # Create text-based summary
    cell_text = []
    for q in questions:
        row = []
        for model in models:
            row.append(dominant_answers[q][model][:40])
        cell_text.append(row)

    ax.axis('off')
    table = ax.table(cellText=cell_text, rowLabels=q_labels,
                     colLabels=models, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    table.auto_set_column_width(list(range(len(models))))

    ax.set_title("Frontier LLM Response Convergence", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "llm_survey_convergence.png"), dpi=150,
                bbox_inches='tight')
    plt.close()
    print("Saved llm_survey_convergence.png")


def plot_entropy_comparison(analysis_path="results/exp3_analysis.json",
                             save_dir="figures"):
    """Compare response entropy across architectures."""
    os.makedirs(save_dir, exist_ok=True)

    with open(analysis_path) as f:
        analysis = json.load(f)

    entropy_data = analysis.get("entropy", {})
    if not entropy_data:
        print("No entropy data found, skipping")
        return

    models = list(entropy_data.keys())
    mean_entropies = [entropy_data[m]["mean_entropy"] for m in models]
    per_prompt = [entropy_data[m]["per_prompt_entropies"] for m in models]

    # Color by architecture
    colors = []
    for m in models:
        if "transformer" in m:
            colors.append("#2196F3")
        elif "gru" in m:
            colors.append("#FF5722")
        else:
            colors.append("#4CAF50")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart of mean entropy
    bars = axes[0].bar(range(len(models)), mean_entropies, color=colors)
    axes[0].set_xticks(range(len(models)))
    axes[0].set_xticklabels(models, rotation=30, ha='right', fontsize=9)
    axes[0].set_ylabel("Mean Response Entropy (bits)")
    axes[0].set_title("Response Diversity by Architecture")
    axes[0].grid(True, alpha=0.3, axis='y')

    for bar, e in zip(bars, mean_entropies):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{e:.2f}", ha='center', va='bottom', fontsize=9)

    # Box plot of per-prompt entropies
    bp = axes[1].boxplot(per_prompt, labels=models, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[1].set_ylabel("Response Entropy (bits)")
    axes[1].set_title("Per-Prompt Entropy Distribution")
    axes[1].tick_params(axis='x', rotation=30)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "entropy_comparison.png"), dpi=150,
                bbox_inches='tight')
    plt.close()
    print("Saved entropy_comparison.png")


def plot_representation_analysis(rep_path="results/exp4_representation_analysis.json",
                                   save_dir="figures"):
    """Visualize representation-level attractor metrics."""
    os.makedirs(save_dir, exist_ok=True)

    with open(rep_path) as f:
        data = json.load(f)

    comparison = data.get("architecture_comparison", {})
    per_model = data.get("per_model_metrics", {})

    if not comparison:
        print("No comparison data found, skipping")
        return

    archs = list(comparison.keys())
    colors_map = {"transformer": "#2196F3", "gru": "#FF5722", "lstm": "#4CAF50"}
    colors = [colors_map.get(a, "#000") for a in archs]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Within vs Between concept similarity
    within = [comparison[a]["mean_within_similarity"] for a in archs]
    between = [comparison[a]["mean_between_similarity"] for a in archs]

    x = np.arange(len(archs))
    w = 0.35
    axes[0].bar(x - w/2, within, w, label='Within-concept', color=colors, alpha=0.8)
    axes[0].bar(x + w/2, between, w, label='Between-concept', color=colors, alpha=0.4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([a.upper() for a in archs])
    axes[0].set_ylabel("Cosine Similarity")
    axes[0].set_title("Concept Clustering")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # 2. Concept separation
    separations = [comparison[a]["mean_separation"] for a in archs]
    bars = axes[1].bar(range(len(archs)), separations, color=colors)
    axes[1].set_xticks(range(len(archs)))
    axes[1].set_xticklabels([a.upper() for a in archs])
    axes[1].set_ylabel("Separation Score")
    axes[1].set_title("Concept Separation\n(within - between similarity)")
    axes[1].grid(True, alpha=0.3, axis='y')

    # 3. Within-concept variance (attractor basin size)
    variances = [comparison[a]["mean_variance"] for a in archs]
    bars = axes[2].bar(range(len(archs)), variances, color=colors)
    axes[2].set_xticks(range(len(archs)))
    axes[2].set_xticklabels([a.upper() for a in archs])
    axes[2].set_ylabel("Mean L2 Distance from Centroid")
    axes[2].set_title("Attractor Basin Size\n(smaller = tighter attractor)")
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "representation_analysis.png"), dpi=150,
                bbox_inches='tight')
    plt.close()
    print("Saved representation_analysis.png")


def plot_scale_effect(results_path="results/training_results.json",
                       rep_path="results/exp4_representation_analysis.json",
                       save_dir="figures"):
    """Show how scaling affects attractor properties."""
    os.makedirs(save_dir, exist_ok=True)

    with open(results_path) as f:
        train_results = json.load(f)

    with open(rep_path) as f:
        rep_data = json.load(f)

    per_model = rep_data.get("per_model_metrics", {})

    # Extract scale vs separation for each architecture
    fig, ax = plt.subplots(figsize=(8, 5))

    arch_data = {"transformer": [], "gru": [], "lstm": []}
    for model_name, metrics in per_model.items():
        params = train_results.get(model_name, {}).get("num_params", 0)
        sep = metrics.get("concept_separation", 0)

        for arch in arch_data:
            if arch in model_name:
                arch_data[arch].append((params, sep))
                break

    colors_map = {"transformer": "#2196F3", "gru": "#FF5722", "lstm": "#4CAF50"}
    markers = {"transformer": "s", "gru": "o", "lstm": "^"}

    for arch, points in arch_data.items():
        if points:
            params, seps = zip(*sorted(points))
            ax.plot(params, seps, '-o', color=colors_map[arch],
                    marker=markers[arch], markersize=8, label=arch.upper(),
                    linewidth=2)

    ax.set_xlabel("Number of Parameters")
    ax.set_ylabel("Concept Separation Score")
    ax.set_title("Scale Effect on Attractor Formation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "scale_effect.png"), dpi=150,
                bbox_inches='tight')
    plt.close()
    print("Saved scale_effect.png")


def run_statistical_tests(analysis_path="results/exp3_analysis.json",
                           rep_path="results/exp4_representation_analysis.json"):
    """Run statistical tests comparing architectures."""
    print("\n" + "=" * 60)
    print("STATISTICAL TESTS")
    print("=" * 60)

    stat_results = {}

    # Test 1: Entropy differences
    if os.path.exists(analysis_path):
        with open(analysis_path) as f:
            analysis = json.load(f)

        entropy_data = analysis.get("entropy", {})

        # Compare transformer vs GRU entropy distributions
        for arch_pair in [("transformer", "gru"), ("transformer", "lstm"), ("gru", "lstm")]:
            a1, a2 = arch_pair
            e1 = []
            e2 = []
            for model, data in entropy_data.items():
                if a1 in model and "medium" not in model:
                    e1 = data["per_prompt_entropies"]
                elif a2 in model and "medium" not in model:
                    e2 = data["per_prompt_entropies"]

            if e1 and e2:
                t_stat, p_val = stats.mannwhitneyu(e1, e2, alternative='two-sided')
                effect = abs(np.mean(e1) - np.mean(e2)) / (np.std(e1 + e2) + 1e-8)
                result = {
                    "test": "Mann-Whitney U",
                    "statistic": float(t_stat),
                    "p_value": float(p_val),
                    "effect_size_cohens_d": float(effect),
                    "mean_1": float(np.mean(e1)),
                    "mean_2": float(np.mean(e2)),
                    "significant": p_val < 0.05,
                }
                stat_results[f"entropy_{a1}_vs_{a2}"] = result
                print(f"\n  Entropy {a1.upper()} vs {a2.upper()}:")
                print(f"    Mean {a1}: {np.mean(e1):.3f}, Mean {a2}: {np.mean(e2):.3f}")
                print(f"    U-statistic: {t_stat:.2f}, p-value: {p_val:.4f}")
                print(f"    Cohen's d: {effect:.3f}")
                print(f"    Significant: {p_val < 0.05}")

    # Test 2: Representation separation differences
    if os.path.exists(rep_path):
        with open(rep_path) as f:
            rep_data = json.load(f)

        per_model = rep_data.get("per_model_metrics", {})

        # Compare concept separation scores
        for model_name, metrics in per_model.items():
            within = list(metrics.get("within_concept_similarity", {}).values())
            between = list(metrics.get("between_concept_similarity", {}).values())

            if within and between:
                t_stat, p_val = stats.mannwhitneyu(within, between, alternative='greater')
                result = {
                    "test": "Mann-Whitney U (within > between)",
                    "model": model_name,
                    "statistic": float(t_stat),
                    "p_value": float(p_val),
                    "mean_within": float(np.mean(within)),
                    "mean_between": float(np.mean(between)),
                    "significant": p_val < 0.05,
                }
                stat_results[f"separation_{model_name}"] = result
                print(f"\n  Concept separation for {model_name}:")
                print(f"    Within: {np.mean(within):.4f}, Between: {np.mean(between):.4f}")
                print(f"    p-value: {p_val:.4f}, Significant: {p_val < 0.05}")

    # Save statistical results
    def convert_types(obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return obj

    with open("results/statistical_tests.json", "w") as f:
        json.dump(stat_results, f, indent=2, default=convert_types)

    print("\nStatistical results saved to results/statistical_tests.json")
    return stat_results


def main():
    """Generate all visualizations and run statistical tests."""
    os.makedirs("figures", exist_ok=True)

    if os.path.exists("results/training_results.json"):
        plot_training_curves()
        plot_perplexity_comparison()

    if os.path.exists("results/exp1_llm_survey.json"):
        plot_llm_survey_convergence()

    if os.path.exists("results/exp3_analysis.json"):
        plot_entropy_comparison()

    if os.path.exists("results/exp4_representation_analysis.json"):
        plot_representation_analysis()

    if (os.path.exists("results/training_results.json") and
        os.path.exists("results/exp4_representation_analysis.json")):
        plot_scale_effect()

    run_statistical_tests()


if __name__ == "__main__":
    main()
