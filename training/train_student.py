"""
LoRA fine-tune of the text-to-SQL student on data/text2sql (harness/gen_sql_data.py export).

Runs inside the PyTorch container with envs/train, via slurm/train_student.slurm. Loss is on
the SQL only (prompt-completion format). Writes the adapter to OUT-adapter and the merged
bf16 model to OUT, which vLLM serves like any other model (slurm/eval_student.slurm).
Checkpoints go to OUT-checkpoints every --save-steps steps and are never deleted. A run
continues from the latest checkpoint there automatically (from scratch if there is none), so
after a crash just resubmit with the same arguments. --resume STEP (or a checkpoint path)
starts from an earlier checkpoint instead; later checkpoints are then overwritten as training
passes their steps again. For a genuinely fresh run use a new --out.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import trl.kernels.chunked_logprob
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

# The container's torch 2.10 pre-release rejects addmm(..., out_dtype=float32) with an fp32
# accumulator and bf16 inputs; use TRL's plain fp32 path instead (same result, a bit slower).
trl.kernels.chunked_logprob._MM_OUT_DTYPE = False

PROJECT_DIR = Path(os.environ.get("P7_PROJECT_DIR", "/ceph/project/P7-Agent-Harness"))


def to_prompt_completion(row: dict) -> dict:
    return {"prompt": row["messages"][:-1], "completion": row["messages"][-1:]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    p.add_argument("--base", type=Path, default=PROJECT_DIR / "models/qwen2.5-coder-1.5b-instruct")
    p.add_argument("--data", type=Path, default=PROJECT_DIR / "data/text2sql")
    p.add_argument("--out", type=Path, default=PROJECT_DIR / "models/qwen2.5-coder-1.5b-sql-v1")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--max-steps", type=int, default=-1, help="for smoke tests")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--save-steps", type=int, default=100, help="checkpoint interval (~15 min)")
    p.add_argument("--resume", metavar="STEP_OR_PATH",
                   help="start from this earlier checkpoint instead of the latest one "
                        "(later checkpoints get overwritten)")
    args = p.parse_args()
    checkpoint_dir = str(args.out) + "-checkpoints"
    if args.resume:
        resume_from = (args.resume if "/" in args.resume
                       else f"{checkpoint_dir}/checkpoint-{int(args.resume)}")
        if not Path(resume_from).is_dir():
            raise SystemExit(f"--resume: {resume_from} does not exist")
    elif Path(checkpoint_dir).is_dir():
        resume_from = get_last_checkpoint(checkpoint_dir)  # None if no checkpoint yet
    else:
        resume_from = None
    print(f"resuming from {resume_from}" if resume_from else "starting from scratch", flush=True)

    data = load_dataset("json", data_files={
        "train": str(args.data / "train.jsonl"), "val": str(args.data / "val.jsonl"),
    })
    data = data.map(to_prompt_completion, remove_columns=data["train"].column_names)

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16)

    config = SFTConfig(
        output_dir=checkpoint_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,  # a float is a fraction of the total steps
        bf16=True,
        gradient_checkpointing=True,
        max_length=2048,
        completion_only_loss=True,  # loss on the SQL, not the schema prompt
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=args.save_steps,  # no save_total_limit: old checkpoints are never deleted
        report_to="none",
        seed=0,
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=data["train"],
        eval_dataset=data["val"],
        processing_class=tokenizer,
        peft_config=LoraConfig(
            r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
            target_modules="all-linear", task_type="CAUSAL_LM",
        ),
    )
    start = time.time()
    trainer.train(resume_from_checkpoint=resume_from)
    metrics = trainer.evaluate()

    trainer.save_model(str(args.out) + "-adapter")
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    info = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "train_examples": len(data["train"]),
        "val_examples": len(data["val"]),
        "final_eval": metrics,
        "minutes": round((time.time() - start) / 60, 1),
    }
    (args.out / "train_info.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
