import glob

import torch
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

BATCH_SIZE = [128, 256, 512, 1024, 2048]
NUM_EPOCHS = 3

# def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
#     vllm_set_random_seed(seed)
    
#     world_size_patch = patch("torch.distributed.get_world_size", return_value=1)


def main():
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Math-1.5B", 
        torch_dtype=torch.float16, 
        device_map="auto",
    )
    
    train_dataset = load_dataset("json", data_files=sorted(glob.glob("data/dart_math/train/*.jsonl")), split="train")
    
    
    for _ in range(NUM_EPOCHS):
        pass
    
    
        
    

if __name__ == "__main__":
    main()
