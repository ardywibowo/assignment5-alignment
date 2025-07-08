import glob

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.sft import (
    get_response_log_probs,
    sft_microbatch_train_step,
    tokenize_prompt_and_output,
)

# BATCH_SIZE = [128, 256, 512, 1024]
BATCH_SIZE = 128
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5

# def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
#     vllm_set_random_seed(seed)
    
#     world_size_patch = patch("torch.distributed.get_world_size", return_value=1)


def main():
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Math-1.5B", 
        torch_dtype=torch.float16, 
        device_map="auto",
    )
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
    train_dataset = load_dataset("json", data_files=sorted(glob.glob("data/dart_math/train/*.jsonl")), split="train")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()
    
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            prompts = batch["prompt"]
            outputs = batch["output"]
            
            tokenized = tokenize_prompt_and_output(prompts, outputs, tokenizer)
            
            input_ids = tokenized["input_ids"]
            labels = tokenized["labels"]
            response_mask = tokenized["response_mask"]
            
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                log_prob_results = get_response_log_probs(model, input_ids, labels)
                policy_log_probs = log_prob_results["log_probs"]
                
                loss, metadata = sft_microbatch_train_step(
                    policy_log_probs=policy_log_probs,
                    response_mask=response_mask,
                    gradient_accumulation_steps=1
                )
            
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")
    
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")

if __name__ == "__main__":
    main()
