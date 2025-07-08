import json
from pathlib import Path
from typing import Callable, Optional
from vllm import LLM, SamplingParams


def load_math_validation_data(data_path: str | Path = "data/MATH/validation.jsonl") -> list[dict]:
    """Load MATH validation dataset from JSONL file.
    
    Args:
        data_path: Path to the validation.jsonl file
        
    Returns:
        List of problem dictionaries with keys: problem, level, type, solution, subject, file_id
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"MATH validation file not found at {data_path}")
    
    problems = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    
    print(f"Loaded {len(problems)} problems from {data_path}")
    return problems


def load_prompt_template(template_path: str | Path = "cs336_alignment/prompts/r1_zero.prompt") -> str:
    """Load prompt template from file.
    
    Args:
        template_path: Path to the prompt template file
        
    Returns:
        Template string with {question} placeholder
    """
    template_path = Path(template_path)
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found at {template_path}")
    
    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()
    
    return template


def format_math_problems(problems: list[dict], template: str) -> list[str]:
    """Format MATH problems using the prompt template.
    
    Args:
        problems: List of MATH problem dictionaries
        template: Prompt template string with {question} placeholder
        
    Returns:
        List of formatted prompts
    """
    prompts = []
    for problem in problems:
        prompt = template.format(question=problem['problem'])
        prompts.append(prompt)
    
    return prompts


def evaluate_llm(
    vllm_model: LLM,
    reward_fn: Optional[Callable[[str, str], dict[str, float]]],
    prompts: list[str],
    eval_sampling_params: SamplingParams,
) -> None:
    responses = vllm_model.generate(prompts, eval_sampling_params)
    for prompt, response in zip(prompts, responses):
        response_text = response.outputs[0].text.strip()
        
        if reward_fn:
            metrics = reward_fn(prompt, response_text)
            
        print(f"Prompt: {prompt}")
        print(f"Response: {response_text}")
        
        if reward_fn:
            print(f"Metrics: {metrics}")
        
        print("-" * 80)

if __name__ == "__main__":
    # Load MATH validation data
    problems = load_math_validation_data()
    
    # Load prompt template
    template = load_prompt_template()
    
    # Format problems with template
    prompts = format_math_problems(problems, template)
    
    import ipdb
    ipdb.set_trace()
    
    # Initialize model
    vllm_model = LLM(model="Qwen/Qwen2.5-Math-1.5B")
    
    # Example: evaluate on first 5 problems
    sample_prompts = prompts[:5]
    print(f"Sample prompt:\n{sample_prompts[0]}")
    print("-" * 80)
    
    sampling_params = SamplingParams(
        temperature = 1.0,
        top_p = 1.0,
        max_tokens = 1024,
        stop = "</answer>",
        include_stop_str_in_output = True,
    )
