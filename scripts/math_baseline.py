from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.eval import (
    evaluate_llm,
    extract_answers_from_problems,
    format_math_problems,
    load_math_validation_data,
    load_prompt_template,
)


def main():
    # Load MATH validation data
    problems = load_math_validation_data()

    # Load prompt template
    template = load_prompt_template()

    # Format problems with template
    prompts = format_math_problems(problems, template)

    # Extract ground truth answers
    answers = extract_answers_from_problems(problems)

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
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop="</answer>",
        include_stop_str_in_output=True,
    )

    results = evaluate_llm(
        vllm_model,
        r1_zero_reward_fn,
        prompts,
        answers,
        sampling_params,
        output_path="artifacts/eval_results.json",
    )

    print("\nEvaluation completed!")
    print(f"Aggregated metrics: {results['aggregated_metrics']}")


if __name__ == "__main__":
    main()
