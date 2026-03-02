"""
Evaluate Llama 3.1 8B zero-shot performance on GSM8K.

Loads GSM8K test data, formats zero-shot prompts, generates outputs via vLLM,
parses predicted answers (last number in output), and writes results to disk.

Usage:
    # Full evaluation
    python scripts/evaluate_gsm8k.py

    # Analyze pre-computed results
    python scripts/evaluate_gsm8k.py --analyze-only --results-path outputs/gsm8k_eval_results.json

    # Custom model
    python scripts/evaluate_gsm8k.py --model-path meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from statistics import mean

from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).parent.parent))
from tests.adapters import run_parse_gsm8k_response

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "meta-llama/Llama-3.1-8B"
DEFAULT_DATA_PATH = Path(__file__).parent.parent / "data" / "gsm8k" / "test.jsonl"
DEFAULT_OUTPUT_PATH = Path(__file__).parent.parent / "outputs" / "gsm8k_eval_results.json"
PROMPT_TEMPLATE_PATH = Path(__file__).parent.parent / "cs336_alignment" / "prompts" / "zero_shot_system_prompt.prompt"


def load_gsm8k_data(data_path: str) -> list[dict]:
    """Load GSM8K examples from JSONL. Each line: {"question": ..., "answer": ...}"""
    examples = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            # Extract the numeric answer after ####
            answer_text = ex["answer"]
            match = re.search(r"####\s*(.+)$", answer_text)
            numeric_answer = match.group(1).strip().replace(",", "") if match else answer_text
            examples.append({
                "question": ex["question"],
                "answer_text": answer_text,
                "numeric_answer": numeric_answer,
            })
    logger.info(f"Loaded {len(examples)} examples from {data_path}")
    return examples


def format_gsm8k_prompt(example: dict, prompt_template: str) -> str:
    """Format a GSM8K question as a zero-shot prompt."""
    instruction = (
        f"Solve the following math problem step by step.\n\n"
        f"Question: {example['question']}\n\n"
        f"Answer:"
    )
    return prompt_template.format(instruction=instruction)


def run_inference(
    model_path: str,
    examples: list[dict],
    prompt_template: str,
    num_gpus: int,
    max_tokens: int,
    temperature: float,
) -> list[dict]:
    """Run vLLM inference on GSM8K examples and return result dicts."""
    prompts = [format_gsm8k_prompt(ex, prompt_template) for ex in examples]

    logger.info(f"Running inference on {len(prompts)} examples with model {model_path}")

    model = LLM(
        model=model_path,
        tensor_parallel_size=num_gpus,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1.0,
    )
    raw_outputs = model.generate(prompts, sampling_params)

    results = []
    for output, example in zip(raw_outputs, examples):
        model_output = output.outputs[0].text
        parsed = run_parse_gsm8k_response(model_output=model_output)
        correct = parsed == example["numeric_answer"] if parsed else False
        results.append({
            "question": example["question"],
            "answer_text": example["answer_text"],
            "numeric_answer": example["numeric_answer"],
            "model_output": model_output,
            "parsed": parsed,
            "correct": correct,
        })

    return results


def save_results(results: list[dict], output_path: str) -> None:
    """Write results and accuracy summary to JSON."""
    overall = round(mean(r["correct"] for r in results), 4)

    data = {
        "results": results,
        "accuracy": {"overall": overall},
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Wrote {len(results)} results to {output_path}")


def load_results(results_path: str) -> dict:
    """Load results from a previously saved JSON file."""
    with open(results_path) as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data['results'])} results from {results_path}")
    return data


def analyze(results: list[dict]) -> None:
    """Print accuracy summary and example predictions."""
    overall = mean(r["correct"] for r in results)
    n_parsed = sum(1 for r in results if r["parsed"] is not None)

    print(f"\n{'='*70}")
    print(f"  GSM8K Zero-Shot Evaluation Results")
    print(f"{'='*70}")
    print(f"Total examples:   {len(results)}")
    print(f"Parsed outputs:   {n_parsed} ({100 * n_parsed / len(results):.1f}%)")
    print(f"Overall accuracy: {100 * overall:.2f}%")

    correct_examples = [r for r in results if r["correct"]]
    wrong_examples = [r for r in results if not r["correct"] and r["parsed"]]
    unparsed_examples = [r for r in results if r["parsed"] is None]

    for label, examples in [
        ("Correct predictions", correct_examples),
        ("Wrong predictions (parsed)", wrong_examples),
        ("Unparsed outputs", unparsed_examples),
    ]:
        print(f"\n{'='*70}")
        print(f"  {label} ({len(examples)} total, showing up to 3)")
        print(f"{'='*70}")
        for i, ex in enumerate(examples[:3]):
            print(f"\n--- Example {i+1} ---")
            print(f"Q: {ex['question'][:200]}")
            print(f"Expected: {ex['numeric_answer']}  |  Parsed: {ex['parsed']}  |  Correct: {ex['correct']}")
            print(f"Output: {ex['model_output'][:300]}")


def main(args) -> None:
    if args.analyze_only:
        data = load_results(args.results_path)
        analyze(data["results"])
    else:
        examples = load_gsm8k_data(args.data_path)
        if not examples:
            logger.error("No examples loaded. Check --data-path.")
            return

        prompt_template = PROMPT_TEMPLATE_PATH.read_text()
        results = run_inference(
            model_path=args.model_path,
            examples=examples,
            prompt_template=prompt_template,
            num_gpus=args.num_gpus,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        save_results(results, args.output_path)
        analyze(results)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser(
        description="Evaluate zero-shot GSM8K performance with vLLM."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH,
                        help=f"Model path or HF ID (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH),
                        help=f"Path to GSM8K test JSONL (default: {DEFAULT_DATA_PATH})")
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH),
                        help=f"Path to write results JSON (default: {DEFAULT_OUTPUT_PATH})")
    parser.add_argument("--results-path", default=str(DEFAULT_OUTPUT_PATH),
                        help="Path to pre-computed results for --analyze-only")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Skip inference, just analyze existing results")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="Number of GPUs for tensor parallelism (default: 1)")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max tokens to generate per example (default: 512)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0 for greedy)")
    args = parser.parse_args()
    logger.info("running %s", " ".join(sys.argv))
    main(args)
