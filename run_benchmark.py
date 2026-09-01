#!/usr/bin/env python3
"""
Run VLM Benchmark on Dataset
"""

import asyncio
import argparse
import json
import os
import sys
import wandb
from pathlib import Path
from typing import List, Dict, Optional
import traceback
from datetime import datetime, timezone

# Add current directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vlm_interaction import create_vlm, VLMInteractionSystem
from environment import PlaywrightEnv


def normalize_answer(answer: str) -> str:
    """Normalize answer string for comparison."""
    if not answer:
        return "None"

    answer = answer.lower().strip()
    if "true" in answer:
        return "True"
    if "false" in answer:
        return "False"
    if "not enough info" in answer or "notenoughinfo" in answer:
        return "NotEnoughInfo"
    return answer


def get_resume_state(
    wandb_entity: str, wandb_project: str, run_id: str
) -> tuple[int, str, int, int]:
    """
    Look up the last completed sample number in an existing W&B run and return:
      (next_offset, run_name, existing_num_samples, existing_passed_count)
    Assumes `sample_number` is logged once per completed sample (as this script does).
    """
    api = wandb.Api()
    run = api.run(f"{wandb_entity}/{wandb_project}/runs/{run_id}")

    seen_samples = set()
    passed_count = 0
    max_sample = None
    for row in run.scan_history(keys=["sample_number", "passed"]):
        sn = row.get("sample_number", None)
        if sn is None:
            continue
        try:
            sn_i = int(sn)
        except Exception:
            continue
        if sn_i in seen_samples:
            continue
        seen_samples.add(sn_i)
        if row.get("passed") is True:
            passed_count += 1
        if max_sample is None or sn_i > max_sample:
            max_sample = sn_i

    next_offset = 0 if max_sample is None else max_sample + 1
    run_name = getattr(run, "name", None) or run_id
    return next_offset, run_name, len(seen_samples), passed_count


async def main():
    parser = argparse.ArgumentParser(description="Run Visworld Benchmark")

    # Dataset args
    parser.add_argument(
        "--dataset", required=True, help="Path to JSONL dataset"
    )
    parser.add_argument(
        "--html-root", required=True, help="Root directory for HTML files"
    )
    parser.add_argument(
        "--output-dir", default="output", help="Directory for outputs"
    )
    parser.add_argument(
        "--limit", type=int, help="Limit number of samples to run"
    )
    parser.add_argument(
        "--offset", type=int, default=0, help="Offset to start from"
    )

    # WandB args
    parser.add_argument(
        "--wandb-project",
        default="insight-benchmark",
        help="WandB project name",
    )
    parser.add_argument(
        "--wandb-entity", help="WandB entity name (defaults to your W&B default entity)", default=None
    )
    parser.add_argument("--run-name", help="WandB run name")
    parser.add_argument(
        "--continue",
        dest="continue_run",
        action="store_true",
        help="Resume an existing W&B run by auto-detecting the last finished sample and continuing from there.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="W&B run id to resume (required with --continue).",
    )

    # VLM args
    parser.add_argument(
        "--model",
        choices=["openai", "huggingface"],
        default="huggingface",
        help="VLM model to use (huggingface or openai)",
    )
    parser.add_argument("--api-key", help="API key for OpenAI/Gemini")
    parser.add_argument("--model-name", help="Model name for OpenAI/Gemini API")
    parser.add_argument("--model-path", help="Path to HuggingFace model")
    parser.add_argument("--base-url", help="Base URL for OpenAI/Gemini API")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retries for OpenAI/Gemini API",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["minimal", "low", "medium", "high"],
        help="Reasoning effort for OpenAI/Gemini API",
        default="high",
    )

    # HuggingFace generation params
    parser.add_argument(
        "--do-sample",
        dest="do_sample",
        action="store_true",
        default=True,
        help="Enable sampling for local HuggingFace model (default: true)",
    )
    parser.add_argument(
        "--no-do-sample",
        dest="do_sample",
        action="store_false",
        help="Disable sampling for local HuggingFace model",
    )
    parser.add_argument("--top-p", type=float, default=0.95, help="top_p")
    parser.add_argument("--top-k", type=int, default=20, help="top_k")
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="repetition_penalty",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="temperature"
    )
    parser.add_argument(
        "--default-context-length",
        type=int,
        default=None,
        help="Fallback context window size if model config does not expose it",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="If set, overrides dynamic context-window-based max_new_tokens",
    )

    # Interaction args
    parser.add_argument(
        "--parse-mode",
        choices=["default", "ui_tars"],
        default="default",
        help="Which action parser to use (default uses <action>...</action>; ui_tars uses Thought/Action with start_box syntax).",
    )
    parser.add_argument(
        "--prompt-prefix",
        default="",
        help="Prefix for prompt files (e.g. 'ui_tars_' to use ui_tars_user_first_interaction.txt).",
    )
    parser.add_argument(
        "--max-turns", type=int, default=10, help="Max turns per sample"
    )
    parser.add_argument(
        "--headless", action="store_true", help="Run headless", default=True
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save screenshots",
        default=False,
    )

    args = parser.parse_args()

    # Initialize WandB
    wandb.login()

    if args.continue_run:
        if not args.run_id:
            print("Error: --run-id is required when using --continue")
            sys.exit(1)

        try:
            (
                resume_offset,
                existing_run_name,
                existing_num_samples,
                existing_passed,
            ) = get_resume_state(
                wandb_entity=args.wandb_entity,
                wandb_project=args.wandb_project,
                run_id=args.run_id,
            )
            print(
                f"Detected run {args.run_id} ({existing_run_name}) with last sample_number {resume_offset}, continuing from there."
            )
            # For local outputs, default to the existing run's name so resumed jobs
            # write into the same output/<run_name>/... directory.
            if args.run_name is None:
                args.run_name = existing_run_name
            # Don't allow resuming to move backwards if user passed an explicit offset.
            if resume_offset > args.offset:
                print(
                    f"Resuming run {args.run_id} ({existing_run_name}): detected last sample_number={resume_offset - 1}, "
                    f"setting --offset to {resume_offset}"
                )
                args.offset = resume_offset
            else:
                print(
                    f"Resuming run {args.run_id} ({existing_run_name}): detected resume offset {resume_offset}, "
                    f"but keeping user --offset {args.offset}"
                )
        except Exception as e:
            print(
                f"Error: failed to detect resume state from W&B run {args.run_id}: {e}.\n"
                f"Refusing to continue because --continue was set (to avoid duplicating samples)."
            )
            sys.exit(1)

    if args.run_name is None:
        model_path = (
            os.path.basename(args.model_path.rstrip("/"))
            if args.model_path
            else args.model_path
        )

        model_name = args.model_name if args.model_name else model_path
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_UTC")

        args.run_name = (
            f"{args.model}-{model_name}-{args.max_turns}-{timestamp}"
        )

    # wandb config is args without API key
    wandb_config = args.__dict__.copy()
    if "api_key" in wandb_config:
        del wandb_config["api_key"]

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.run_name if not args.continue_run else None,
        id=args.run_id if args.continue_run else None,
        resume="allow" if args.continue_run else None,
        config=wandb_config,
    )

    # Create output directory in output_dir/run_name
    output_dir = Path(args.output_dir) / args.run_name
    print(f"Output directory: {output_dir}")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load dataset
    samples = []
    with open(args.dataset, "r") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    # Apply limit/offset
    samples = samples[args.offset :]
    if args.limit:
        samples = samples[: args.limit]

    print(f"Loaded {len(samples)} samples to run")

    # Setup Model
    print("Initializing Model...")
    vlm_kwargs = {}
    if args.api_key:
        vlm_kwargs["api_key"] = args.api_key
    if args.model_name:
        vlm_kwargs["model_name"] = args.model_name
    if args.model_path:
        vlm_kwargs["model_path"] = args.model_path
    if args.base_url:
        vlm_kwargs["base_url"] = args.base_url
    if args.max_retries:
        vlm_kwargs["max_retries"] = int(args.max_retries)
    if args.reasoning_effort:
        vlm_kwargs["reasoning_effort"] = args.reasoning_effort
    if args.model == "huggingface":
        vlm_kwargs["do_sample"] = args.do_sample
        vlm_kwargs["top_p"] = args.top_p
        vlm_kwargs["top_k"] = args.top_k
        vlm_kwargs["repetition_penalty"] = args.repetition_penalty
        vlm_kwargs["temperature"] = args.temperature
        vlm_kwargs["default_context_length"] = args.default_context_length
        vlm_kwargs["max_new_tokens"] = args.max_new_tokens
        if args.prompt_prefix:
            vlm_kwargs["prompt_prefix"] = args.prompt_prefix
    try:
        vlm = create_vlm(args.model, **vlm_kwargs)
    except Exception as e:
        print(f"Error creating VLM: {e}")
        sys.exit(1)

    # Setup Environment
    print("Initializing Environment...")
    env = PlaywrightEnv(headless=args.headless)
    await env.setup()

    try:
        results_table = wandb.Table(
            columns=[
                "sample_number",
                "source_id",
                "html_file",
                "proposition",
                "passed",
                "model_answer",
                "parsed_model_answer",
                "label",
                "num_interactions",
                "max_interactions",
                "model_actions",
                "reasoning_tokens",
                "completion_tokens",
                "total_tokens",
                "termination_reason",
                "parse_error_turn",
                "failed",
                "error",
            ]
        )

        # If continuing a run, seed counts so final accuracy reflects the full run.
        passed_count = existing_passed if args.continue_run else 0
        processed_count = existing_num_samples if args.continue_run else 0
        last_logged_step = None

        for i, sample in enumerate(samples):
            # Construct HTML path
            html_rel_path = sample["html_file"]
            html_path = os.path.join(args.html_root, html_rel_path)

            if not os.path.exists(html_path):
                # Log and continue; don't crash whole run
                log_data = {
                    "sample_number": i + args.offset,
                    "source_id": sample.get("source_id"),
                    "html_file": html_rel_path,
                    "proposition": sample.get("proposition"),
                    "passed": False,
                    "model_answer": None,
                    "parsed_model_answer": "None",
                    "label": sample.get("class"),
                    "num_interactions": 0,
                    "max_interactions": args.max_turns,
                    "model_actions": "[]",
                    "reasoning_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "termination_reason": "html_not_found",
                    "parse_error_turn": None,
                    "failed": True,
                    "error": f"HTML file not found: {html_path}",
                }
                # Use sample_number as the W&B step to keep steps aligned across resumes.
                wandb.log(log_data, step=int(log_data["sample_number"]))
                last_logged_step = int(log_data["sample_number"])
                results_table.add_data(
                    log_data["sample_number"],
                    log_data["source_id"],
                    log_data["html_file"],
                    log_data["proposition"],
                    log_data["passed"],
                    log_data["model_answer"],
                    log_data["parsed_model_answer"],
                    log_data["label"],
                    log_data["num_interactions"],
                    log_data["max_interactions"],
                    log_data["model_actions"],
                    log_data["reasoning_tokens"],
                    log_data["completion_tokens"],
                    log_data["total_tokens"],
                    log_data["termination_reason"],
                    log_data["parse_error_turn"],
                    log_data["failed"],
                    log_data["error"],
                )
                continue

            # Reset environment for new sample
            await env.reset()

            # Create Interaction System for this sample
            sample_output_dir = output_dir / sample["source_id"]

            system = VLMInteractionSystem(
                vlm=vlm,
                env=env,
                max_turns=args.max_turns,
                save_images=args.save_images,
                output_dir=str(sample_output_dir),
                parse_mode=args.parse_mode,
            )

            error_msg = None
            failed = False
            termination_reason = None
            parse_error_turn = None
            result = {}

            try:
                result = await system.run_interaction(
                    html_file=os.path.abspath(html_path),
                    proposition=sample["proposition"],
                )
            except Exception as e:
                failed = True
                termination_reason = "runner_exception"
                error_msg = str(e)
                parse_error_turn = None

            # Process results
            model_answer_raw = result.get("answer") if result else None
            model_answer = normalize_answer(model_answer_raw)
            label = normalize_answer(sample["class"])

            passed = model_answer == label
            if passed:
                passed_count += 1
            processed_count += 1

            history = result.get("history", [])
            num_interactions = len(history)
            model_actions = [entry["action"] for entry in history]

            reasoning_tokens = result.get("thinking_tokens", 0)
            completion_tokens = result.get("completion_tokens", 0)
            total_tokens = result.get("total_tokens", 0)

            # Prefer structured outcome from run_interaction (covers parse errors and API failures)
            if result:
                termination_reason = (
                    termination_reason
                    or result.get("failure_reason")
                    or ("answered" if model_answer_raw else None)
                )
                parse_error_turn = result.get("parse_error_turn")
                error_msg = error_msg or result.get("error")
                failed = failed or bool(result.get("failed", False))

            # Log to WandB
            log_data = {
                "sample_number": i + args.offset,
                "source_id": sample["source_id"],
                "html_file": html_rel_path,
                "proposition": sample["proposition"],
                "passed": passed,
                "model_answer": model_answer_raw,
                "parsed_model_answer": model_answer,
                "label": sample["class"],
                "num_interactions": num_interactions,
                "max_interactions": args.max_turns,
                "model_actions": str(
                    model_actions
                ),  # Convert list to string for flat logging
                "reasoning_tokens": reasoning_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "termination_reason": termination_reason,
                "parse_error_turn": parse_error_turn,
                "failed": failed,
                "error": error_msg,
            }

            # Use sample_number as the W&B step to keep steps aligned across resumes.
            wandb.log(log_data, step=int(log_data["sample_number"]))
            last_logged_step = int(log_data["sample_number"])
            results_table.add_data(
                i + args.offset,
                sample["source_id"],
                html_rel_path,
                sample["proposition"],
                passed,
                model_answer_raw,
                model_answer,
                sample["class"],
                num_interactions,
                args.max_turns,
                str(model_actions),
                reasoning_tokens,
                completion_tokens,
                total_tokens,
                termination_reason,
                parse_error_turn,
                failed,
                error_msg,
            )

        # Log final table.
        # On resume, don't overwrite the original table key; log a chunk table keyed by offset.
        table_key = "results_table"
        if args.continue_run:
            table_key = f"results_table_from_{args.offset}"
        if last_logged_step is None:
            # No samples were logged in this process; avoid forcing a step.
            wandb.log({table_key: results_table})
        else:
            wandb.log({table_key: results_table}, step=int(last_logged_step))

        accuracy = passed_count / processed_count if processed_count else 0
        print(f"\nFinal Accuracy: {accuracy:.2%}")
        if last_logged_step is None:
            wandb.log({"accuracy": accuracy})
        else:
            wandb.log({"accuracy": accuracy}, step=int(last_logged_step))

    finally:
        await env.cleanup()
        wandb.finish()


if __name__ == "__main__":
    asyncio.run(main())
