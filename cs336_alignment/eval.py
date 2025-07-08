import json
from pathlib import Path
from typing import Callable, Optional, Union, Any

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
    ground_truths: list[Any],
    eval_sampling_params: SamplingParams,
    output_path: Optional[str] = None,
) -> dict:
    """Evaluate LLM on prompts and calculate metrics.
    
    Args:
        vllm_model: The VLLM model to evaluate
        reward_fn: Function to calculate reward metrics
        prompts: List of input prompts
        ground_truths: List of ground truth answers
        eval_sampling_params: Sampling parameters for generation
        output_path: Optional path to save results
        
    Returns:
        Dictionary containing evaluation results and aggregated metrics
    """
    responses = vllm_model.generate(prompts, eval_sampling_params)
    
    results = []
    all_metrics = []
    
    for i, (prompt, response, ground_truth) in enumerate(zip(prompts, responses, ground_truths)):
        response_text = response.outputs[0].text.strip()
        
        # Calculate metrics if reward function provided
        metrics = {}
        if reward_fn:
            metrics = reward_fn(response_text, ground_truth)
            all_metrics.append(metrics)
        
        # Store individual result
        result = {
            "id": i,
            "prompt": prompt,
            "response": response_text,
            "ground_truth": ground_truth,
            "metrics": metrics
        }
        results.append(result)
        
        # Print progress
        print(f"Example {i+1}/{len(prompts)}")
        print(f"Response: {response_text}")
        if metrics:
            print(f"Metrics: {metrics}")
        print("-" * 80)
    
    # Calculate aggregated metrics
    aggregated_metrics = {}
    if all_metrics:
        metric_keys = all_metrics[0].keys()
        for key in metric_keys:
            values = [m[key] for m in all_metrics if key in m]
            if values:
                aggregated_metrics[f"avg_{key}"] = sum(values) / len(values)
                aggregated_metrics[f"total_{key}"] = sum(values)
    
    # Prepare final results
    eval_results = {
        "examples": results,
        "aggregated_metrics": aggregated_metrics,
        "num_examples": len(results)
    }
    
    # Save to disk if path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        print(f"Results saved to {output_path}")
    
    return eval_results

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

    results = evaluate_llm(
        vllm_model, 
        r1_zero_reward_fn, 
        sample_prompts, 
        sample_answers, 
        sampling_params,
        output_path="eval_results.json"
    )
    
    print("\nEvaluation completed!")
    print(f"Aggregated metrics: {results['aggregated_metrics']}")
