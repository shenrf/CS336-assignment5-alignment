"""
SFT instruction-tuning training.

Usage:
    # Basic training (uses data/sft-instruct/train.jsonl by default)
    uv run python scripts/train_sft.py

    # Use the smaller sample dataset
    uv run python scripts/train_sft.py --train-data data/sft-instruct/sample_train.jsonl

    # Custom hyperparameters
    uv run python scripts/train_sft.py \
        --lr 1e-5 \
        --epochs 3 \
        --batch-size 4 \
        --gradient-accumulation-steps 8
"""

import argparse
from pathlib import Path

from cs336_alignment.sft import SFTTrainer

DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-Math-1.5B"
DEFAULT_TRAIN_DATA = Path(__file__).parent.parent / "data" / "sft-reason" / "sft_gpt-oss-120b_filtered.jsonl"
DEFAULT_VAL_DATA = Path(__file__).parent.parent / "data" / "sft-reason" / "val.jsonl"
DEFAULT_PROMPT_TEMPLATE = Path(__file__).parent.parent / "data" / "sft-reason" / "r1_zero.prompt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT instruction-tuning training")

    # Data
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--train-data", default=str(DEFAULT_TRAIN_DATA))
    parser.add_argument("--val-data", default=str(DEFAULT_VAL_DATA))
    parser.add_argument("--prompt-template", default=str(DEFAULT_PROMPT_TEMPLATE))
    parser.add_argument("--output-dir", default="outputs/sft")

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--flash-attn", action="store_true", default=True)
    parser.add_argument("--no-flash-attn", dest="flash_attn", action="store_false")

    # Logging and evaluation
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=0, help="Eval every N steps. 0 to disable.")
    parser.add_argument("--max-val-examples", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--no-wandb", dest="wandb", action="store_false")
    parser.add_argument("--wandb-project", default="cs336-sft")
    parser.add_argument("--run-name", default=None)

    args = parser.parse_args()

    trainer = SFTTrainer(
        model_path=args.model_path,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        prompt_template_path=args.prompt_template,
        output_dir=args.output_dir,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        device=args.device,
        flash_attn=args.flash_attn,
        log_every=args.log_every,
        eval_every=args.eval_every,
        max_val_examples=args.max_val_examples,
        save_every=args.save_every,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
    )
    trainer.train()
