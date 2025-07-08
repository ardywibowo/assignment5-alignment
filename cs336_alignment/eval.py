from typing import Callable
from vllm import LLM, SamplingParams


def evaluate_llm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    eval_sampling_params: SamplingParams,
) -> None:
    responses = vllm_model.generate(prompts, eval_sampling_params)
    for prompt, response in zip(prompts, responses):
        response_text = response.outputs[0].text.strip()
        metrics = reward_fn(prompt, response_text)
        print(f"Prompt: {prompt}")
        print(f"Response: {response_text}")
        print(f"Metrics: {metrics}")
        print("-" * 80)

if __name__ == "__main__":
    
    vllm_model = LLM(model="Qwen/Qwen2.5-Math-1.5B")
