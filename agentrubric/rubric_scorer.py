"""
Rubric scorer — Phase 1 LangChain scoring chain.

Takes a task + model response + rubric text, calls the Groq LLM to grade
each rubric criterion, and returns a structured RubricResult with per-criterion
scores and a weighted overall score.

This is the foundation that the rest of the system builds on:
  - reward_scorer_node (agents/reward_scorer.py) wraps score_response()
    as a LangGraph node so the graph can call it on both rubric variants
  - gold_judge_node (eval/gold_judge.py) is a separate independent scorer
    that deliberately does NOT use this file — it scores without a rubric
    to provide an unbiased ground-truth signal for hack detection

Core functions:
  load_rubric(path)                     — reads rubric text from a .txt file
  parse_rubric_criteria(rubric_text)    — extracts criterion names and weights
  build_scoring_prompt(task, response, rubric_text) — constructs the LLM prompt
  call_llm(prompt)                      — calls Groq Llama 3.1 8B, returns raw string
  parse_llm_output(raw)                 — extracts criteria_scores list from JSON
  score_response(sample_id, task, response, rubric_text) — full pipeline, returns RubricResult

Data models:
  CriterionScore — name, weight, score (0.0–1.0), reasoning (one sentence)
  RubricResult   — list of CriterionScore + weighted overall_score + passed flag
                   passed = True if overall_score >= PASS_THRESHOLD
"""

import argparse
import json
import re
import sys
from pathlib import Path

from agentrubric.config import get_llm, GROQ_API_KEY
from langchain_core.prompts import HumanMessagePromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel
from groq import RateLimitError, AuthenticationError
from agentrubric.constants import PASS_THRESHOLD
from agentrubric.utils import truncate_task, truncate_reasoning
from agentrubric.logger import get_logger

logger = get_logger(__name__)

PACKAGE_ROOT = Path(__file__).parent
PROJECT_ROOT = Path(__file__).parent.parent
RUBRICS_DIR = PACKAGE_ROOT / "rubrics"
DATA_DIR = PROJECT_ROOT / "data"



class CriterionScore(BaseModel):
    name: str
    weight: float
    score: float
    reasoning: str


class RubricResult(BaseModel):
    sample_id: str
    task: str
    response: str
    criteria: list[CriterionScore]
    overall_score: float
    passed: bool


def load_rubric(path: str) -> str:
    """Load rubric template from a text file."""
    try:
        return Path(path).read_text(encoding='utf-8').strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Rubric file not found: {path}")


def parse_rubric_criteria(rubric_text: str) -> list[dict]:
    """Parse CRITERIA section and extract name and weight for each criterion."""
    criteria = []
    pattern = r'^\d+\.\s+([^(]+?)\s*\(weight:\s*([\d.]+)\)'

    for line in rubric_text.split('\n'):
        match = re.match(pattern, line)
        if match:
            name = match.group(1).strip()
            weight = float(match.group(2))
            criteria.append({"name": name, "weight": weight})

    if not criteria:
        raise ValueError("No criteria found in rubric text")

    return criteria


def build_scoring_prompt(task: str, response: str, rubric_text: str) -> str:
    """Build the scoring prompt with task, response, and rubric."""
    criteria_list = parse_rubric_criteria(rubric_text)
    example_criteria = ",\n    ".join(
        f'{{"name": "{c["name"]}", "score": 0.8, "reasoning": "one sentence"}}'
        for c in criteria_list
    )

    prompt = f"""You are an expert evaluator. Score the following model response against the rubric.

RUBRIC:
{rubric_text}

TASK GIVEN TO THE MODEL:
{task}

MODEL RESPONSE:
{response}

INSTRUCTIONS:
Score each criterion from 0.0 to 1.0 where:
  0.0 = completely fails the criterion
  0.5 = partially meets the criterion
  1.0 = fully meets the criterion

Return ONLY a valid JSON object in this exact format, no extra text, no markdown:
{{
  "criteria_scores": [
    {example_criteria}
  ]
}}"""
    return prompt


def call_llm(prompt: str) -> str:
    """Call Groq LLM with the prompt and return raw output."""
    try:
        logger.debug("LLM call prompt length: %d chars", len(prompt))
        llm = get_llm(temperature=0)
        prompt_template = PromptTemplate(template="{prompt}", input_variables=["prompt"])
        chain = prompt_template | llm | StrOutputParser()
        result = chain.invoke({"prompt": prompt})
        return result
    except RateLimitError as e:
        logger.error("LLM rate limit: %s", e)
        raise RuntimeError("Groq rate limit hit — wait 60s and retry")
    except AuthenticationError as e:
        logger.error("LLM authentication: %s", e)
        raise RuntimeError("Invalid GROQ_API_KEY — check your .env file")
    except Exception as e:
        logger.error("LLM call failed: %s: %s", type(e).__name__, e)
        raise RuntimeError(f"LLM call failed: {type(e).__name__}: {e}")


def parse_llm_output(raw_output: str) -> list[dict]:
    """Parse LLM output and extract criteria_scores list."""
    output = raw_output.strip()

    if "```json" in output:
        start = output.find("```json") + 7
        end = output.find("```", start)
        output = output[start:end].strip()
    elif "```" in output:
        start = output.find("```") + 3
        end = output.find("```", start)
        output = output[start:end].strip()

    try:
        data = json.loads(output)
        return data.get("criteria_scores", [])
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Failed to parse LLM output: {raw_output[:200]}")


def score_response(sample_id: str, task: str, response: str, rubric_text: str) -> RubricResult:
    """Score a model response against a rubric using an LLM."""
    if not GROQ_API_KEY:
        raise EnvironmentError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")

    try:
        criteria_metadata = parse_rubric_criteria(rubric_text)

        prompt = build_scoring_prompt(task, response, rubric_text)
        raw_output = call_llm(prompt)
        scored_criteria_list = parse_llm_output(raw_output)

        criteria = []
        overall_score = 0.0

        for scored in scored_criteria_list:
            metadata = next((c for c in criteria_metadata if c["name"] == scored["name"]), None)
            if metadata:
                weight = metadata["weight"]
                score = scored["score"]
                criterion = CriterionScore(
                    name=scored["name"],
                    weight=weight,
                    score=score,
                    reasoning=scored["reasoning"]
                )
                criteria.append(criterion)
                overall_score += score * weight

        overall_score = round(overall_score, 4)
        passed = overall_score >= PASS_THRESHOLD

        return RubricResult(
            sample_id=sample_id,
            task=task,
            response=response,
            criteria=criteria,
            overall_score=overall_score,
            passed=passed
        )
    except Exception as e:
        error_msg = str(e)
        logger.warning("score_response failed for %s: %s", sample_id, error_msg)
        criteria_metadata = parse_rubric_criteria(rubric_text)
        criteria = [
            CriterionScore(
                name=c["name"],
                weight=c["weight"],
                score=0.0,
                reasoning=f"ERROR: {error_msg}"
            )
            for c in criteria_metadata
        ]
        return RubricResult(
            sample_id=sample_id,
            task=task,
            response=response,
            criteria=criteria,
            overall_score=0.0,
            passed=False
        )


def print_result(result: RubricResult) -> None:
    """Print formatted rubric result with color support."""
    use_color = sys.stdout.isatty()
    GREEN = "\033[92m" if use_color else ""
    RED = "\033[91m" if use_color else ""
    RESET = "\033[0m" if use_color else ""

    status = "[PASS]" if result.passed else "[FAIL]"
    status_colored = f"{GREEN}{status}{RESET}" if result.passed else f"{RED}{status}{RESET}"

    task_truncated = truncate_task(result.task)

    print("-" * 50)
    print(f"Sample ID : {result.sample_id}")
    print(f"Task      : {task_truncated}")
    print("-" * 50)
    print(f"Overall score : {result.overall_score:.4f}  {status_colored}")
    print()
    print("Criteria breakdown:")

    for criterion in result.criteria:
        name = criterion.name
        weight = criterion.weight
        score = criterion.score
        reasoning_truncated = truncate_reasoning(criterion.reasoning)

        print(f"  {name:<15} (x{weight:.2f})  ->  {score:.2f}   \"{reasoning_truncated}\"")

    print("-" * 50)


def main():
    """Score all samples and save results to JSON."""
    parser = argparse.ArgumentParser(description="Score model responses against a rubric")
    parser.add_argument("--rubric", default=str(RUBRICS_DIR / "default.txt"), help="Path to rubric file")
    parser.add_argument("--data", default=str(DATA_DIR / "sample_responses.json"), help="Path to sample responses JSON")
    parser.add_argument("--output", default=str(DATA_DIR / "results.json"), help="Path to save results JSON")
    args = parser.parse_args()

    if not Path(args.rubric).exists():
        print(f"ERROR: Rubric file not found: {args.rubric}", file=sys.stderr)
        sys.exit(1)

    if not Path(args.data).exists():
        print(f"ERROR: Data file not found: {args.data}", file=sys.stderr)
        sys.exit(1)

    rubric_text = load_rubric(args.rubric)

    try:
        with open(args.data, "r") as f:
            samples = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in data file: {args.data}", file=sys.stderr)
        print(f"  {e}", file=sys.stderr)
        sys.exit(1)

    results = []
    total_count = len(samples)
    passed_count = 0
    failed_count = 0
    total_score = 0.0

    for idx, sample in enumerate(samples, 1):
        sample_id = sample["id"]
        print(f"Scoring {sample_id} ({idx}/{total_count})...")

        try:
            result = score_response(
                sample_id=sample_id,
                task=sample["task"],
                response=sample["response"],
                rubric_text=rubric_text
            )
            results.append(result)
            print_result(result)

            if result.passed:
                passed_count += 1
            else:
                failed_count += 1
            total_score += result.overall_score

        except Exception as e:
            print(f"  ERROR: {str(e)}")
            print()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump([r.model_dump() for r in results], f, indent=2)

    print()
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total samples     : {total_count}")
    print(f"Samples scored    : {len(results)}")
    print(f"Passed (>= {PASS_THRESHOLD}) : {passed_count}")
    print(f"Failed (< {PASS_THRESHOLD})  : {failed_count}")
    if results:
        print(f"Average score     : {total_score / len(results):.4f}")
    print()
    print(f"Results saved to {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
