#!/usr/bin/env python3
"""
VLM Web Interaction System
Multi-turn interaction with HTML pages using VLMs.
"""

import os
import sys
import asyncio
import argparse
from typing import Optional, List, Dict
from pathlib import Path
from PIL import Image, ImageDraw
import io

from environment import PlaywrightEnv
from models import BaseVLM, OpenAIModel, HuggingFaceModel


class VLMInteractionSystem:
    """Multi-turn VLM interaction system for HTML pages."""

    def __init__(
        self,
        vlm: BaseVLM,
        env: PlaywrightEnv,
        max_turns: int = 10,
        save_images: bool = False,
        output_dir: str = "output",
        parse_mode: str = "default",
    ):
        """
        Initialize the interaction system.

        Args:
            vlm: Vision Language Model instance
            env: Playwright environment instance
            max_turns: Maximum number of interaction turns
            save_images: Whether to save screenshots
            output_dir: Directory to save outputs
        """
        self.vlm = vlm
        self.env = env
        self.max_turns = max_turns
        self.save_images = save_images
        self.output_dir = Path(output_dir)
        self.interaction_history: List[Dict] = []
        self.parse_mode = parse_mode

        if save_images:
            self.output_dir.mkdir(exist_ok=True)

    def save_screenshot(
        self,
        screenshot_bytes: bytes,
        filename: str,
        action: Optional[dict] = None,
    ):
        """Save screenshot with optional action visualization."""
        if not self.save_images:
            return

        image = Image.open(io.BytesIO(screenshot_bytes))

        if action:
            draw = ImageDraw.Draw(image)

            if action["type"] in ["click", "hover", "shift_click"]:
                x, y = action["x"], action["y"]
                radius = 15
                color_mapping = {
                    "click": "red",
                    "hover": "blue",
                    "shift_click": "green",
                }
                color = color_mapping[action["type"]]

                # Draw circle
                draw.ellipse(
                    [x - radius, y - radius, x + radius, y + radius],
                    fill=color,
                    outline="dark" + color,
                    width=3,
                )

                # Draw crosshair
                draw.line(
                    [x - radius * 2, y, x + radius * 2, y], fill=color, width=2
                )
                draw.line(
                    [x, y - radius * 2, x, y + radius * 2], fill=color, width=2
                )
            if action["type"] == "drag":
                start_pos = action["start_pos"]
                end_pos = action["end_pos"]
                color = "purple"
                draw.line(
                    [start_pos[0], start_pos[1], end_pos[0], end_pos[1]],
                    fill=color,
                    width=2,
                )

        filepath = self.output_dir / filename
        image.save(filepath)

    async def run_interaction(self, html_file: str, proposition: str) -> Dict:
        """
        Run multi-turn interaction with HTML page.

        Maintains full conversation history with both text and images:
        1. System prompt + initial screenshot + proposition
        2. Assistant response with action
        3. Updated screenshot + follow-up prompt
        4. Loop continues until answer or max turns

        Args:
            html_file: Path to HTML file
            proposition: proposition to answer

        Returns:
            Dict containing answer, stats, and history
        """
        thinking_tokens = 0
        completion_tokens = 0
        prompt_tokens = 0
        total_tokens = 0

        # Outcome metadata for dataset-level logging
        failed = False  # True only for unexpected/system errors (API failures, playwright errors, etc.)
        failure_reason = None  # e.g. "parse_error", "max_turns", "vlm_query_error", "env_execute_error"
        error = None
        last_response = None
        parse_error_turn = None

        # Load HTML
        try:
            await self.env.load_html(html_file)
            await self.env.wait(1000)  # Wait for page to settle
        except Exception as e:
            failed = True
            failure_reason = "env_load_error"
            error = str(e)
            return {
                "answer": None,
                "failed": failed,
                "failure_reason": failure_reason,
                "error": error,
                "last_response": last_response,
                "parse_error_turn": parse_error_turn,
                "thinking_tokens": thinking_tokens,
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "total_tokens": total_tokens,
                "history": self.interaction_history,
            }

        # Take initial screenshot
        screenshot_bytes = await self.env.screenshot()
        if self.save_images:
            self.save_screenshot(screenshot_bytes, "turn_0_initial.png")

        answer = None
        # Conversation history with multimodal content (text + images)
        # Format: [{"role": "...", "content": [{"type": "text"|"image", ...}]}]
        conversation_history = []

        # Add system message at the start
        # conversation_history.append(
        #     {
        #         "role": "system",
        #         "content": [{"type": "text", "text": self.vlm.system_prompt}],
        #     }
        # )

        for turn in range(1, self.max_turns + 1):
            # Build current user message with screenshot and appropriate prompt
            screenshot_image = Image.open(io.BytesIO(screenshot_bytes))

            if turn == 1:
                # First turn - use initial interaction prompt with proposition
                user_text = self.vlm.user_first_interaction.replace(
                    "[PROPOSITION]", proposition
                )

                user_text = user_text.replace(
                    "[MAX_TURNS]", str(self.max_turns)
                )

            else:
                # Follow-up turns - use update prompt
                user_text = self.vlm.user_update_prompt.replace(
                    "[PROPOSITION]", proposition
                )

            # Add current user message to conversation history
            conversation_history.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": screenshot_image},
                        {"type": "text", "text": user_text},
                    ],
                }
            )

            # Query VLM with complete conversation history (including current turn)
            try:
                response_data = self.vlm.query(conversation_history)
                response = response_data["text"]
                extra_content = response_data.get("extra_content", {})
            except Exception as e:
                failed = True
                failure_reason = "vlm_query_error"
                error = str(e)
                return {
                    "answer": None,
                    "failed": failed,
                    "failure_reason": failure_reason,
                    "error": error,
                    "last_response": last_response,
                    "parse_error_turn": parse_error_turn,
                    "thinking_tokens": thinking_tokens,
                    "completion_tokens": completion_tokens,
                    "prompt_tokens": prompt_tokens,
                    "total_tokens": total_tokens,
                    "history": self.interaction_history,
                }

            last_response = response

            if "thinking_tokens" in response_data:
                thinking_tokens += response_data.get("thinking_tokens", 0)
                completion_tokens += response_data.get("completion_tokens", 0)
                prompt_tokens += response_data.get("prompt_tokens", 0)
                total_tokens += response_data.get("total_tokens", 0)

            # Extract action
            if self.parse_mode == "ui_tars":
                action = self.vlm.extract_action_ui_tars(response)
            else:
                action = self.vlm.extract_action(response)

            if not action:
                failure_reason = "parse_error"
                error = f"Could not parse action from response. Response: {response}"
                parse_error_turn = turn
                return {
                    "answer": None,
                    "failed": failed,
                    "failure_reason": failure_reason,
                    "thinking_tokens": thinking_tokens,
                    "completion_tokens": completion_tokens,
                    "prompt_tokens": prompt_tokens,
                    "total_tokens": total_tokens,
                    "history": self.interaction_history,
                    "error": error,
                    "last_response": last_response,
                    "parse_error_turn": parse_error_turn,
                }

            new_assistant_msg = {
                "role": "assistant",
                "content": [{"type": "text", "text": response}],
            }

            if extra_content:
                new_assistant_msg["extra_content"] = extra_content

            # Add assistant's response to conversation history
            conversation_history.append(new_assistant_msg)

            # Store in interaction history (for summary)
            self.interaction_history.append(
                {
                    "turn": turn,
                    "response": response,
                    "action": action,
                }
            )

            # Check if we have an answer
            if action["type"] == "answer":
                answer = action["value"]
                break

            # Execute action
            try:
                await self.env.execute_action(action)
                await self.env.wait(500)
            except Exception as e:
                failed = True
                failure_reason = "env_execute_error"
                error = str(e)
                return {
                    "answer": None,
                    "failed": failed,
                    "failure_reason": failure_reason,
                    "error": error,
                    "last_response": last_response,
                    "parse_error_turn": parse_error_turn,
                    "thinking_tokens": thinking_tokens,
                    "completion_tokens": completion_tokens,
                    "prompt_tokens": prompt_tokens,
                    "total_tokens": total_tokens,
                    "history": self.interaction_history,
                }

            # Take screenshot after action
            screenshot_bytes = await self.env.screenshot()
            if self.save_images:
                filename = f"turn_{turn}_{action['type']}.png"
                self.save_screenshot(screenshot_bytes, filename, action)

        if not answer:
            if failure_reason is None:
                failure_reason = "max_turns"

        return {
            "answer": answer,
            "failed": failed,
            "failure_reason": failure_reason,
            "error": error,
            "last_response": last_response,
            "parse_error_turn": parse_error_turn,
            "thinking_tokens": thinking_tokens,
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "total_tokens": total_tokens,
            "history": self.interaction_history,
        }

    def print_summary(self, answer: Optional[str]):
        """Print summary of interaction."""
        print(f"\n{'='*80}")
        print(f"SUMMARY")
        print(f"{'='*80}")
        print(f"Total Turns: {len(self.interaction_history)}")
        print(f"\nInteraction History:")
        for entry in self.interaction_history:
            action = entry["action"]
            action_desc = self.env.get_action_description(action)
            print(f"  Turn {entry['turn']}: {action_desc}")

        if answer:
            print(f"\n✅ Final Answer: {answer}")
        else:
            print(f"\n❌ No answer found")
        print(f"{'='*80}\n")


def create_vlm(model_type: str, **kwargs) -> BaseVLM:
    """Create VLM instance based on model type."""
    if model_type == "openai":
        if "api_key" not in kwargs:
            raise ValueError("api_key is required for OpenAI/Gemini model")

        # Extract OpenAI/Gemini parameters
        openai_kwargs = {"api_key": kwargs["api_key"]}
        if "model_name" in kwargs and kwargs["model_name"]:
            openai_kwargs["model"] = kwargs["model_name"]
        if "base_url" in kwargs and kwargs["base_url"]:
            openai_kwargs["base_url"] = kwargs["base_url"]
        if "max_retries" in kwargs and kwargs["max_retries"]:
            openai_kwargs["max_retries"] = kwargs["max_retries"]
        if "reasoning_effort" in kwargs and kwargs["reasoning_effort"]:
            openai_kwargs["reasoning_effort"] = kwargs["reasoning_effort"]
        return OpenAIModel(**openai_kwargs)

    elif model_type == "huggingface":
        if "model_path" not in kwargs:
            raise ValueError("model_path is required for HuggingFace model")
        hf_kwargs = {"model_path": kwargs["model_path"]}
        for k in [
            "do_sample",
            "top_p",
            "top_k",
            "repetition_penalty",
            "temperature",
            "default_context_length",
            "max_new_tokens",
            "prompt_prefix",
        ]:
            if k in kwargs and kwargs[k] is not None:
                hf_kwargs[k] = kwargs[k]
        return HuggingFaceModel(**hf_kwargs)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="VLM Web Interaction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("html_file", help="Path to HTML file")
    parser.add_argument("proposition", help="proposition to answer")
    parser.add_argument(
        "--model",
        choices=["openai", "huggingface", "local"],
        default="huggingface",
        help="VLM model to use",
    )
    parser.add_argument("--api-key", help="API key for OpenAI/Gemini")
    parser.add_argument(
        "--model-name",
        help="Model name (e.g., gpt-4o, gemini-2.0-flash, gemini-2.5-pro)",
    )
    parser.add_argument("--model-path", help="Path to HuggingFace model")
    parser.add_argument(
        "--base-url",
        help="Base URL for API (use for Gemini: https://generativelanguage.googleapis.com/v1beta/openai/)",
    )
    parser.add_argument(
        "--max-turns", type=int, default=10, help="Maximum interaction turns"
    )
    parser.add_argument(
        "--headless", action="store_true", help="Run browser in headless mode"
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Don't save screenshots"
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory for screenshots",
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

    args = parser.parse_args()

    # Validate arguments
    if args.model == "openai" and not args.api_key:
        print("Error: --api-key is required for OpenAI model")
        sys.exit(1)

    if not os.path.exists(args.html_file):
        print(f"Error: HTML file not found: {args.html_file}")
        sys.exit(1)

    # Create VLM
    vlm_kwargs = {}
    if args.api_key:
        vlm_kwargs["api_key"] = args.api_key
    if args.model_name:
        vlm_kwargs["model_name"] = args.model_name
    if args.model_path:
        vlm_kwargs["model_path"] = args.model_path
    if args.base_url:
        vlm_kwargs["base_url"] = args.base_url
    if args.model == "huggingface":
        vlm_kwargs["do_sample"] = args.do_sample
        vlm_kwargs["top_p"] = args.top_p
        vlm_kwargs["top_k"] = args.top_k
        vlm_kwargs["repetition_penalty"] = args.repetition_penalty
        vlm_kwargs["temperature"] = args.temperature
        vlm_kwargs["default_context_length"] = args.default_context_length
        vlm_kwargs["max_new_tokens"] = args.max_new_tokens

    try:
        vlm = create_vlm(args.model, **vlm_kwargs)
    except Exception as e:
        print(f"Error creating VLM: {e}")
        sys.exit(1)

    # Create environment
    env = PlaywrightEnv(headless=args.headless)
    await env.setup()

    # Create interaction system
    system = VLMInteractionSystem(
        vlm=vlm,
        env=env,
        max_turns=args.max_turns,
        save_images=not args.no_save,
        output_dir=args.output_dir,
    )

    try:
        # Run interaction
        result = await system.run_interaction(args.html_file, args.proposition)
        answer = result["answer"]

        # Print summary
        system.print_summary(answer)

        if answer:
            sys.exit(0)
        else:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        await env.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
