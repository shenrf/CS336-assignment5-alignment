"""
Evaluate Llama 3.1 8B zero-shot performance on MMLU.

Loads MMLU test CSVs, formats zero-shot prompts, generates outputs via vLLM,
parses predictions, and writes per-example results + accuracy to disk.

Usage:
    # Full evaluation
    python scripts/evaluate_mmlu.py

    # Single subject
    python scripts/evaluate_mmlu.py --subjects abstract_algebra

    # Analyze pre-computed results
    python scripts/evaluate_mmlu.py --analyze-only --results-path outputs/mmlu_eval_results.json

    # Custom model
    python scripts/evaluate_mmlu.py --model-path meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

from vllm import LLM, SamplingParams

# Import the MMLU parser from the adapters
sys.path.insert(0, str(Path(__file__).parent.parent))
from tests.adapters import run_parse_mmlu_response

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "meta-llama/Llama-3.1-8B"
DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "mmlu" / "test"
DEFAULT_OUTPUT_PATH = Path(__file__).parent.parent / "outputs" / "mmlu_eval_results.json"
PROMPT_TEMPLATE_PATH = Path(__file__).parent.parent / "cs336_alignment" / "prompts" / "zero_shot_system_prompt.prompt"

LETTER_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}


def load_mmlu_data(data_dir: str, subjects: list[str] | None = None) -> list[dict]:
    """Load MMLU examples from CSV files (no header).

    CSV format: question, option_A, option_B, option_C, option_D, answer_letter
    """
    data_path = Path(data_dir)
    examples = []

    csv_files = sorted(data_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    for csv_file in csv_files:
        # Extract subject from filename: e.g. "abstract_algebra_test.csv" -> "abstract_algebra"
        subject = csv_file.stem
        for suffix in ("_test", "_dev", "_val"):
            subject = subject.removesuffix(suffix)

        if subjects and subject not in subjects:
            continue

        with open(csv_file, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 6:
                    continue
                question = row[0]
                options = [row[1], row[2], row[3], row[4]]
                answer = row[5].strip()
                examples.append({
                    "subject": subject,
                    "question": question,
                    "options": options,
                    "answer": answer,
                })

    logger.info(f"Loaded {len(examples)} examples from {len(csv_files)} files in {data_dir}")
    return examples


def format_mmlu_prompt(example: dict) -> str:
    """Format a single MMLU example as a zero-shot prompt."""
    subject_display = example["subject"].replace("_", " ")
    options = example["options"]
    instruction = (
        f"The following is a multiple choice question about {subject_display}.\n\n"
        f"Question: {example['question']}\n"
        f"A. {options[0]}\n"
        f"B. {options[1]}\n"
        f"C. {options[2]}\n"
        f"D. {options[3]}\n\n"
        f"Answer:"
    )
    return instruction


def format_with_system_prompt(instruction: str, prompt_template: str) -> str:
    """Wrap the instruction in the zero-shot system prompt template."""
    return prompt_template.format(instruction=instruction)


def run_inference(
    model_path: str,
    examples: list[dict],
    prompt_template: str,
    num_gpus: int,
    max_tokens: int,
    temperature: float,
) -> list[dict]:
    """Run vLLM inference on MMLU examples and return result dicts."""
    prompts = []
    for ex in examples:
        instruction = format_mmlu_prompt(ex)
        prompt = format_with_system_prompt(instruction, prompt_template)
        prompts.append(prompt)

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
        parsed = run_parse_mmlu_response(
            mmlu_example=example,
            model_output=model_output,
        )
        correct = parsed == example["answer"] if parsed else False
        results.append({
            "subject": example["subject"],
            "question": example["question"],
            "options": example["options"],
            "answer": example["answer"],
            "model_output": model_output,
            "parsed": parsed,
            "correct": correct,
        })

    return results


def save_results(results: list[dict], output_path: str) -> None:
    """Write results and accuracy summary to JSON."""
    # Per-subject accuracy
    subject_correct = defaultdict(list)
    for r in results:
        subject_correct[r["subject"]].append(r["correct"])

    per_subject = {
        subj: round(mean(scores), 4)
        for subj, scores in sorted(subject_correct.items())
    }
    overall = round(mean(r["correct"] for r in results), 4)

    data = {
        "results": results,
        "accuracy": {
            "overall": overall,
            "per_subject": per_subject,
        },
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
    """Print accuracy summary by subject and overall."""
    subject_correct = defaultdict(list)
    for r in results:
        subject_correct[r["subject"]].append(r["correct"])

    per_subject = {
        subj: mean(scores) for subj, scores in sorted(subject_correct.items())
    }
    overall = mean(r["correct"] for r in results)
    n_parsed = sum(1 for r in results if r["parsed"] is not None)

    print(f"\n{'='*70}")
    print(f"  MMLU Zero-Shot Evaluation Results")
    print(f"{'='*70}")
    print(f"Total examples:  {len(results)}")
    print(f"Parsed outputs:  {n_parsed} ({100 * n_parsed / len(results):.1f}%)")
    print(f"Overall accuracy: {100 * overall:.2f}%")
    print(f"\nPer-subject accuracy ({len(per_subject)} subjects):")
    print(f"{'-'*50}")
    for subj, acc in sorted(per_subject.items(), key=lambda x: -x[1]):
        n = len(subject_correct[subj])
        print(f"  {subj:<45s} {100 * acc:5.1f}% ({n})")

    # Show some example predictions
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
            print(f"\n--- Example {i+1} [{ex['subject']}] ---")
            print(f"Q: {ex['question'][:150]}")
            print(f"Options: A.{ex['options'][0][:30]}  B.{ex['options'][1][:30]}  C.{ex['options'][2][:30]}  D.{ex['options'][3][:30]}")
            print(f"Answer: {ex['answer']}  |  Parsed: {ex['parsed']}  |  Correct: {ex['correct']}")
            print(f"Output: {ex['model_output'][:200]}")


def main(args) -> None:
    if args.analyze_only:
        data = load_results(args.results_path)
        analyze(data["results"])
    else:
        examples = load_mmlu_data(args.data_dir, args.subjects)
        if not examples:
            logger.error("No examples loaded. Check --data-dir and --subjects.")
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
        description="Evaluate zero-shot MMLU performance with vLLM."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH,
                        help=f"Model path or HF ID (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                        help=f"Directory with MMLU test CSVs (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH),
                        help=f"Path to write results JSON (default: {DEFAULT_OUTPUT_PATH})")
    parser.add_argument("--results-path", default=str(DEFAULT_OUTPUT_PATH),
                        help="Path to pre-computed results for --analyze-only")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Skip inference, just analyze existing results")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="Number of GPUs for tensor parallelism (default: 1)")
    parser.add_argument("--max-tokens", type=int, default=64,
                        help="Max tokens to generate per example (default: 64)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0 for greedy)")
    parser.add_argument("--subjects", nargs="*", default=None,
                        help="Specific subjects to evaluate (default: all)")
    args = parser.parse_args()
    logger.info("running %s", " ".join(sys.argv))
    main(args)
