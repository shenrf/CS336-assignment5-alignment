"""
Expert Iteration training on MATH.

Usage:
    uv run python scripts/train_ei.py

    # Custom hyperparameters
    uv run python scripts/train_ei.py \
        --n-ei-steps 3 \
        --questions-per-step 256 \
        --num-generations 8 \
        --sft-epochs 1
"""

import argparse
from pathlib import Path

from cs336_alignment.ei import EITrainer

DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-Math-1.5B"
DEFAULT_TRAIN_DATA = Path(__file__).parent.parent / "data" / "sft-reason" / "sft_gpt-oss-120b.jsonl"
DEFAULT_VAL_DATA = Path(__file__).parent.parent / "data" / "sft-reason" / "val.jsonl"
DEFAULT_PROMPT_TEMPLATE = Path(__file__).parent.parent / "data" / "sft-reason" / "r1_zero.prompt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Expert Iteration training on MATH")

    # Data
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--train-data", default=str(DEFAULT_TRAIN_DATA))
    parser.add_argument("--val-data", default=str(DEFAULT_VAL_DATA))
    parser.add_argument("--prompt-template", default=str(DEFAULT_PROMPT_TEMPLATE))
    parser.add_argument("--output-dir", default="outputs/ei")

    # EI hyperparameters
    parser.add_argument("--n-ei-steps", type=int, default=3)
    parser.add_argument("--questions-per-step", type=int, default=256)
    parser.add_argument("--num-generations", type=int, default=8)

    # Sampling
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--sampling-max-tokens", type=int, default=1024)
    parser.add_argument("--sampling-min-tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # SFT hyperparameters
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--sft-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--flash-attn", action="store_true", default=True)
    parser.add_argument("--no-flash-attn", dest="flash_attn", action="store_false")

    # Logging
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-val-examples", type=int, default=500)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--no-wandb", dest="wandb", action="store_false")
    parser.add_argument("--wandb-project", default="cs336-ei")
    parser.add_argument("--run-name", default=None)

    args = parser.parse_args()

    trainer = EITrainer(
        model_path=args.model_path,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        prompt_template_path=args.prompt_template,
        output_dir=args.output_dir,
        n_ei_steps=args.n_ei_steps,
        questions_per_step=args.questions_per_step,
        num_generations=args.num_generations,
        sampling_temperature=args.sampling_temperature,
        sampling_max_tokens=args.sampling_max_tokens,
        sampling_min_tokens=args.sampling_min_tokens,
        seed=args.seed,
        lr=args.lr,
        weight_decay=args.weight_decay,
        sft_epochs=args.sft_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        device=args.device,
        flash_attn=args.flash_attn,
        log_every=args.log_every,
        max_val_examples=args.max_val_examples,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
    )
    trainer.train()
