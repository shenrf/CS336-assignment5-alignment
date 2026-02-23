"""
Run vLLM inference on the MATH validation set and analyze results.

Loads problems from data/sft-reason/val.jsonl, generates model outputs using
Qwen/Qwen2.5-Math-1.5B (downloaded from HuggingFace), scores with
r1_zero_reward_fn, and writes results to data/sft-reason/baseline_results.jsonl.

Usage:
    # Run inference + analysis
    python scripts/evaluate_math_baseline.py

    # Analyze pre-computed results without running inference
    python scripts/evaluate_math_baseline.py --analyze-only

    # Run with a different model or custom options
    python scripts/evaluate_math_baseline.py \
        --model-path Qwen/Qwen2.5-Math-1.5B-Instruct \
        --num-gpus 2 \
        --max-tokens 4096 \
        --n-examples 5
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from statistics import mean

from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-Math-1.5B"
DEFAULT_VAL_DATA_PATH = Path(__file__).parent.parent / "data" / "sft-reason" / "val.jsonl"
DEFAULT_RESULTS_PATH = Path(__file__).parent.parent / "data" / "sft-reason" / "ai_native_eng_math_results.jsonl"
PROMPT_TEMPLATE_PATH = Path(__file__).parent.parent / "data" / "sft-reason" / "r1_zero.prompt"


def load_val_data(val_data_path: str) -> list[dict]:
    """Load val.jsonl (JSON array of {"problem": ..., "expected_answer": ...})."""
    with open(val_data_path) as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} validation examples from {val_data_path}")
    return data


def run_inference(
    model_path: str,
    val_data: list[dict],
    prompt_template: str,
    num_gpus: int,
    max_tokens: int,
    temperature: float,
) -> list[dict]:
    """Run vLLM inference on val_data and return result dicts."""
    prompts = [prompt_template.format(question=ex["problem"]) for ex in val_data]

    model = LLM(
        model=model_path,
        tensor_parallel_size=num_gpus,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, top_p=1.0)
    # Stop when the model completes its answer; include the stop string in the output.
    # https://github.com/sail-sg/understand-r1-zero/blob/c18804602b85da9e88b4aeeb6c43e2f08c594fbc/train_zero_math.py#L167
    sampling_params.stop = ["</answer>"]
    sampling_params.include_stop_str_in_output = True
    raw_outputs = model.generate(prompts, sampling_params)

    results = []
    for output, example in zip(raw_outputs, val_data):
        generation = output.outputs[0].text
        scores = r1_zero_reward_fn(generation, str(example["expected_answer"]))
        results.append({
            "problem": example["problem"],
            "expected_answer": example["expected_answer"],
            "output": generation,
            "reward": {
                "reward": scores["reward"],
                "format_reward": scores["format_reward"],
                "answer_reward": scores["answer_reward"],
            },
        })

    return results


def save_results(results: list[dict], output_path: str) -> None:
    """Write results in ai_native_eng_math_results.jsonl format (JSON object with results + accuracy)."""
    avg_acc = mean(r["reward"]["answer_reward"] for r in results)
    avg_format_acc = mean(r["reward"]["format_reward"] for r in results)
    data = {
        "results": results,
        "accuracy": {
            "avg_acc": round(avg_acc, 4),
            "avg_format_acc": round(avg_format_acc, 4),
        },
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Wrote {len(results)} results to {output_path}")


def load_results(results_path: str) -> list[dict]:
    """Load results from a file in ai_native_eng_math_results.jsonl format."""
    with open(results_path) as f:
        data = json.load(f)
    results = data["results"]
    if "accuracy" in data:
        logger.info(f"Reported accuracy summary: {data['accuracy']}")
    logger.info(f"Loaded {len(results)} results from {results_path}")
    return results


def categorize(results: list[dict]) -> tuple[list, list, list]:
    """Split results into the three categories."""
    correct, wrong_answer, bad_format = [], [], []
    for r in results:
        fmt = r["reward"]["format_reward"]
        ans = r["reward"]["answer_reward"]
        if fmt == 1.0 and ans == 1.0:
            correct.append(r)
        elif fmt == 1.0 and ans == 0.0:
            wrong_answer.append(r)
        else:
            bad_format.append(r)
    return correct, wrong_answer, bad_format


def print_examples(examples: list[dict], label: str, n: int) -> None:
    print(f"\n{'='*70}")
    print(f"  {label}  ({len(examples)} total, showing {min(n, len(examples))})")
    print(f"{'='*70}")
    for i, ex in enumerate(examples[:n]):
        print(f"\n--- Example {i+1} ---")
        print(f"Problem:  {ex['problem'][:200]}")
        print(f"Expected: {ex['expected_answer']}")
        print(f"Output:   {ex['output'][:400]}")
        print(f"Rewards:  format={ex['reward']['format_reward']}  "
              f"answer={ex['reward']['answer_reward']}")


def analyze(results: list[dict], n_examples: int) -> None:
    correct, wrong_answer, bad_format = categorize(results)
    total = len(results)
    avg_reward = mean(r["reward"]["reward"] for r in results)

    print("\n=== MATH Evaluation Results ===")
    print(f"Total examples : {total}")
    print(
        f"Correct        (fmt=1, ans=1): {len(correct):5d}  "
        f"({100 * len(correct) / total:.1f}%)"
    )
    print(
        f"Wrong answer   (fmt=1, ans=0): {len(wrong_answer):5d}  "
        f"({100 * len(wrong_answer) / total:.1f}%)"
    )
    print(
        f"Bad format     (fmt=0, ans=0): {len(bad_format):5d}  "
        f"({100 * len(bad_format) / total:.1f}%)"
    )
    print(f"Average reward : {avg_reward:.4f}")

    print_examples(correct,      "Category 1: Correct (format=1, answer=1)", n_examples)
    print_examples(wrong_answer, "Category 2: Wrong answer (format=1, answer=0)", n_examples)
    print_examples(bad_format,   "Category 3: Bad format (format=0, answer=0)", n_examples)


def main(args) -> None:
    if args.analyze_only:
        results = load_results(args.results_path)
    else:
        val_data = load_val_data(args.val_data_path)
        prompt_template = PROMPT_TEMPLATE_PATH.read_text()
        results = run_inference(
            model_path=args.model_path,
            val_data=val_data,
            prompt_template=prompt_template,
            num_gpus=args.num_gpus,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        save_results(results, args.output_path)

    analyze(results, args.n_examples)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser(
        description="Run vLLM inference on MATH val set and/or analyze results."
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help=f"Model path or HuggingFace ID for inference (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip inference and analyze --results-path instead.",
    )
    parser.add_argument(
        "--val-data-path",
        default=str(DEFAULT_VAL_DATA_PATH),
        help=f"Path to val.jsonl (default: {DEFAULT_VAL_DATA_PATH})",
    )
    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_RESULTS_PATH),
        help=f"Path to write inference results (default: {DEFAULT_RESULTS_PATH})",
    )
    parser.add_argument(
        "--results-path",
        default=str(DEFAULT_RESULTS_PATH),
        help=f"Path to pre-computed results for analysis (default: {DEFAULT_RESULTS_PATH})",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Max tokens to generate per example (default: 1024)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=10,
        help="Number of examples to print per category (default: 10).",
    )
    args = parser.parse_args()
    logger.info("running %s", " ".join(sys.argv))
    main(args)
