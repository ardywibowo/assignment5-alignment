import json
import re
from pathlib import Path
from typing import Callable, Optional, Union

from math_verify import parse
from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, extract_answer


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


def extract_answers_from_problems(problems: list[dict]) -> list[Union[str, list]]:
    """Extract ground truth answers from MATH problems.
    
    Args:
        problems: List of MATH problem dictionaries
        
    Returns:
        List of extracted answers
    """
    answers = []
    for problem in problems:
        answer = extract_answer(problem['solution'])
        answer = parse(answer)
        answer = normalize_answer(answer)
        
        answers.append(answer)
    
    return answers

def normalize_answer(answer: Union[str, list]) -> Union[str, list]:
    if isinstance(answer, list):
        return [normalize_answer(a) for a in answer]
    else:
        return str(answer)

def evaluate_llm(
    vllm_model: LLM,
    reward_fn: Optional[Callable[[str, str], dict[str, float]]],
    prompts: list[str],
    ground_truths: Optional[list[str]],
    eval_sampling_params: SamplingParams,
) -> None:
    responses = vllm_model.generate(prompts, eval_sampling_params)
    for prompt, response, ground_truth in zip(prompts, responses, ground_truths):
        response_text = response.outputs[0].text.strip()
        
        if reward_fn:
            metrics = reward_fn(response_text, ground_truth)
            
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
    
    # Extract ground truth answers
    answers = extract_answers_from_problems(problems)

    # import ipdb
    # ipdb.set_trace()
    
    # Example: evaluate on first 5 problems
    sample_prompts = prompts[:5]
    sample_answers = answers[:5]
    
    # Initialize model
    vllm_model = LLM(model="Qwen/Qwen2.5-Math-1.5B")
    
    print(f"Sample prompt:\n{sample_prompts[0]}")
    print("-" * 40)
    print(f"Ground truth answer: {sample_answers[0]}")
    print("-" * 80)
    
    sampling_params = SamplingParams(
        temperature = 1.0,
        top_p = 1.0,
        max_tokens = 1024,
        stop = "</answer>",
        include_stop_str_in_output = True,
    )

    evaluate_llm(vllm_model, r1_zero_reward_fn, sample_prompts, sample_answers, sampling_params)
