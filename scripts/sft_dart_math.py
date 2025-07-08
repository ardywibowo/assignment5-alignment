import argparse
import glob
import os
from typing import Optional

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.checkpoint import CheckpointManager
from cs336_alignment.sft import (
    get_response_log_probs,
    sft_microbatch_train_step,
    tokenize_prompt_and_output,
)

# ---------- hyper-params ----------
BATCH_SIZE     = 128
NUM_EPOCHS     = 3
LEARNING_RATE  = 5e-5
SAVE_EVERY     = 500            # optimiser steps
MODEL_NAME     = "Qwen/Qwen2.5-Math-1.5B"
DATA_DIR       = "data/dart_math/train/*.jsonl"
CKPT_DIR       = "models"

# ---------- helper to build the FSDP-wrapped model ----------
def build_fsdp_model(rank: int):
    torch.cuda.set_device(rank)
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,          # loaded in BF16 to skip an extra cast
        device_map={"": rank},               # put weights on *this* GPU
    )

    wrap_policy = transformer_auto_wrap_policy(base_model)
    fsdp_model = FSDP(
        base_model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=rank,
    )
    return fsdp_model

# ---------- the training step ----------
def train_loop(rank: int, world_size: int, resume_from: Optional[str]):
    """One process per GPU."""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.manual_seed(42)

    model = build_fsdp_model(rank)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    dataset = load_dataset("json",
                           data_files=sorted(glob.glob(DATA_DIR)),
                           split="train")
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                         shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    checkpoint_manager  = CheckpointManager(CKPT_DIR, model, optimizer)

    # ── optional resume ────────────────────────────────────────────────────
    start_step = 0
    if resume_from is not None:
        start_step = checkpoint_manager.load(resume_from, strict=False)
        print(f"[rank {rank}] Resumed at global step {start_step}")

    model.train()
    step = start_step
    for epoch in range(NUM_EPOCHS):
        for batch in loader:
            step += 1
            prompts, outputs = batch["prompt"], batch["output"]

            toks = tokenize_prompt_and_output(prompts, outputs, tokenizer)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logprob = get_response_log_probs(
                    model, toks["input_ids"], toks["labels"]
                )["log_probs"]
                loss, _ = sft_microbatch_train_step(
                    policy_log_probs=logprob,
                    response_mask=toks["response_mask"],
                    gradient_accumulation_steps=1,
                )

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if rank == 0 and step % 100 == 0:
                print(f"epoch={epoch} step={step} loss={loss.item():.4f}")

            # -------- periodic async checkpoint (rank 0 schedules it) -------
            if rank == 0 and step % SAVE_EVERY == 0:
                checkpoint_manager.save_async(step)

        # epoch-level sync save so all ranks see the same barrier
        dist.barrier()
        if rank == 0:
            checkpoint_manager.save(step, tag=f"epoch_{epoch}")

    # final checkpoint + consolidate
    dist.barrier()
    if rank == 0:
        checkpoint_manager.save_final(step)

    dist.destroy_process_group()

# ---------- entry-point ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to an existing checkpoint_step_xxx.pth")
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "4"))
    if world_size > 1:
        torch.multiprocessing.spawn(
            train_loop, args=(world_size, args.resume), nprocs=world_size
        )
    else:
        train_loop(0, 1, args.resume)

if __name__ == "__main__":
    main()
