
from vllm import SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

N_EI_STEPS = 5
BATCH_SIZE = [512, 1024, 2048]


# def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
#     vllm_set_random_seed(seed)
    
#     world_size_patch = patch("torch.distributed.get_world_size", return_value=1)


def main():
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        n=G,
        seed=seed,
    )

if __name__ == "__main__":
    main()
