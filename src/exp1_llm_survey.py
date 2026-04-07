"""
Experiment 1: Survey frontier LLMs for default preference responses.
Tests whether transformer-based LLMs converge on similar "persona" answers
(the "tiramisu effect").
"""

import json
import os
import time
from openai import OpenAI

# Use OpenAI API (available via env var)
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

PREFERENCE_QUESTIONS = [
    "What is your favorite dessert?",
    "What color do you prefer?",
    "Which season do you like best?",
    "What is your favorite animal?",
    "What music genre do you prefer?",
    "What time of day do you enjoy most?",
    "Which planet is the most interesting?",
    "What is your favorite number?",
    "What is the best programming language?",
    "What is the best way to learn?",
]

# Models to test via OpenAI-compatible API
MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1-mini",
]

SYSTEM_PROMPT = (
    "Answer the question directly and concisely in 1-2 sentences. "
    "Give your genuine preference or opinion."
)

NUM_SAMPLES = 5  # queries per question per model


def query_model(model: str, question: str, temperature: float = 0.7) -> str:
    """Query a model and return its response."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=temperature,
            max_tokens=100,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {e}"


def run_survey():
    """Run the full survey across models and questions."""
    results = {}

    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"Querying: {model}")
        print(f"{'='*60}")
        results[model] = {}

        for question in PREFERENCE_QUESTIONS:
            print(f"\n  Q: {question}")
            responses = []
            for i in range(NUM_SAMPLES):
                resp = query_model(model, question)
                responses.append(resp)
                print(f"    [{i+1}] {resp[:80]}...")
                time.sleep(0.5)  # rate limiting

            results[model][question] = responses

    # Save results
    output_path = "results/exp1_llm_survey.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


def analyze_survey(results: dict):
    """Analyze convergence patterns across models."""
    print("\n" + "=" * 60)
    print("ANALYSIS: Cross-Model Response Convergence")
    print("=" * 60)

    analysis = {}

    for question in PREFERENCE_QUESTIONS:
        print(f"\nQ: {question}")
        all_responses = []
        model_summaries = {}

        for model in results:
            responses = results[model].get(question, [])
            all_responses.extend(responses)
            # Extract key word (first noun/adjective) from each response
            model_summaries[model] = responses

            # Print first response per model
            if responses:
                print(f"  {model}: {responses[0][:80]}")

        analysis[question] = {
            "responses_by_model": model_summaries,
            "total_responses": len(all_responses),
        }

    # Save analysis
    output_path = "results/exp1_analysis.json"
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nAnalysis saved to {output_path}")

    return analysis


if __name__ == "__main__":
    results = run_survey()
    analysis = analyze_survey(results)
