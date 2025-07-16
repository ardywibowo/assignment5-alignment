#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fine‑tune Qwen2.5‑Math‑1.5B on DART‑Math with FSDP + Torch‑DCP checkpoints.

Launch:
    torchrun --standalone --nnodes 1 --nproc_per_node 4 train_fsdp.py   \
             [--resume /path/to/checkpoint_step_xxx.pth]
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import Optional

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.checkpoint import CheckpointManager
from cs336_alignment.sft import (
    get_response_log_probs,
    sft_microbatch_train_step,
    tokenize_prompt_and_output,
)

# ──────────────────────────────────────────────────────────────────────────
# Hyper‑parameters
# ──────────────────────────────────────────────────────────────────────────
BATCH_SIZE = 4
NUM_EPOCHS = 3
LEARNING_RATE = 1e-5
SAVE_EVERY = 1000  # optimiser steps
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
DATA_DIR = "data/dart_math/train/*.jsonl"
CKPT_DIR = "models"
WARMUP_STEPS = 10  # LR warm‑up steps


# ══════════════════════════════════════════════════════════════════════════
#                               Training
# ══════════════════════════════════════════════════════════════════════════
def build_fsdp_model(rank: int) -> FSDP:
    torch.cuda.set_device(rank)

    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map={"": rank},
    )

    return FSDP(
        base_model,
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=rank,
    )


def build_warmup_scheduler(
    optimizer: torch.optim.Optimizer, warmup_steps: int = WARMUP_STEPS
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        return min((step + 1) / float(warmup_steps), 1.0)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)


# -------------------------------------------------------------------------
def train_loop(rank: int, world_size: int, resume_from: Optional[str]) -> None:
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.manual_seed(42)

    model = build_fsdp_model(rank)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    dataset = load_dataset(
        "json",
        data_files=sorted(glob.glob(DATA_DIR)),
        split="train",
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = build_warmup_scheduler(optimizer)
    ckpt_mgr = CheckpointManager(CKPT_DIR, model, optimizer, scheduler)

    # ── optional resume ───────────────────────────────────────────────────
    start_step = ckpt_mgr.load(resume_from) if resume_from else 0
    if resume_from:
        scheduler.step(start_step)

    # ── main training loop ────────────────────────────────────────────────
    model.train()
    step = start_step
    for epoch in range(NUM_EPOCHS):
        for batch in loader:
            step += 1

            prompts, outputs = batch["query"], batch["response"]
            toks = tokenize_prompt_and_output(prompts, outputs, tokenizer)
            device = torch.device(f"cuda:{rank}")
            toks = {
                k: v.to(device) if torch.is_tensor(v) else v for k, v in toks.items()
            }

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logprob = get_response_log_probs(
                    model, toks["input_ids"], toks["labels"]
                )["log_probs"]

                loss, _ = sft_microbatch_train_step(
                    policy_log_probs=logprob,
                    response_mask=toks["response_mask"],
                    gradient_accumulation_steps=1,
                )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            if rank == 0 and step % 100 == 0:
                print(
                    f"epoch={epoch} step={step} "
                    f"loss={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e}"
                )

            # ── periodic checkpoint (ALL ranks) ──────────────────────────
            if step % SAVE_EVERY == 0:
                ckpt_mgr.save_async(step)

        # epoch‑level checkpoint (ALL ranks) ------------------------------
        ckpt_mgr.save(step, tag=f"epoch_{epoch}")

    # final checkpoint -----------------------------------------------------
    ckpt_mgr.save_final(step)
    dist.destroy_process_group()


# ══════════════════════════════════════════════════════════════════════════
#                               Entrypoint
# ══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume", type=str, default=None, help="Checkpoint to resume from"
    )
    args = parser.parse_args()

    if "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        train_loop(rank, world_size, args.resume)
    else:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size > 1:
            torch.multiprocessing.spawn(
                train_loop, args=(world_size, args.resume), nprocs=world_size
            )
        else:
            train_loop(rank=0, world_size=1, resume_from=args.resume)


if __name__ == "__main__":
    main()
