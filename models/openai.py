import base64
import io
import os
from typing import List, Dict, Optional
from .base import BaseVLM


class OpenAIModel(BaseVLM):
    """OpenAI Vision Language Model implementation."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str = None,
        prompts_dir: str = None,
        prompt_prefix: Optional[str] = None,
        max_retries: int = 5,
        reasoning_effort: str = "high",
    ):
        """
        Initialize OpenAI-compatible model (OpenAI or Gemini).

        Args:
            api_key: API key (OpenAI or Gemini)
            model: Model name (default: gpt-4o)
                   For Gemini: gemini-2.0-flash, gemini-2.5-flash, gemini-2.5-pro, etc.
            base_url: API base URL. If None, uses OpenAI.
                     For Gemini: "https://generativelanguage.googleapis.com/v1beta/openai/"
            prompts_dir: Path to the directory containing prompt files.
                        If None, defaults to 'prompts' directory relative to this file.
            max_retries: Max retries for OpenAI API
            reasoning_effort: Reasoning effort for OpenAI API
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            )

        # Initialize client with optional base_url for Gemini compatibility
        if base_url:
            self.client = OpenAI(
                api_key=api_key, base_url=base_url, max_retries=max_retries
            )
        else:
            self.client = OpenAI(api_key=api_key, max_retries=max_retries)

        self.model = model
        self.reasoning_effort = reasoning_effort
        # Load prompts from files
        if prompts_dir is None:
            # Prompts are one level up from models/ directory
            prompts_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "prompts",
            )

        # self.system_prompt = self._load_prompt_file(
        #     os.path.join(prompts_dir, "system_prompt.txt")
        # )

        prompt_prefix = prompt_prefix or ""
        self.user_first_interaction = self._load_prompt_file(
            os.path.join(
                prompts_dir, f"{prompt_prefix}user_first_interaction.txt"
            )
        )
        # self.user_first_interaction = self._load_prompt_file(
        #     os.path.join(prompts_dir, "user_first_interaction.txt")
        # )
        self.user_update_prompt = self._load_prompt_file(
            os.path.join(prompts_dir, "user_update_prompt.txt")
        )

    @staticmethod
    def _load_prompt_file(filepath: str) -> str:
        """Load a prompt from a text file."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Prompt file not found: {filepath}. "
                f"Please ensure the prompts directory exists with required files."
            )

    @staticmethod
    def _extract_text(content_items: List[Dict]) -> str:
        """Extract text from multimodal content list."""
        for item in content_items:
            if item["type"] == "text":
                return item["text"]
        return ""

    @staticmethod
    def _image_to_base64(pil_image) -> str:
        """Convert PIL Image to base64 string."""
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def query(self, conversation_history: List[Dict]) -> dict:
        """
        Query OpenAI-compatible API with complete conversation history.

        Supports both OpenAI and Gemini APIs via OpenAI SDK.

        Args:
            conversation_history: Complete conversation with current turn's user message.
                Format: [{"role": "...", "content": [{"type": "text"|"image", ...}]}]

        Returns:
            Model response text
        """
        messages = []

        for msg in conversation_history:
            role = msg["role"]

            if role == "system":
                # System and assistant messages are text-only
                text = self._extract_text(msg["content"])
                messages.append({"role": role, "content": text})
            elif role == "assistant":
                # System and assistant messages are text-only
                text = self._extract_text(msg["content"])
                extra_content = msg.get("extra_content", {})
                new_msg = {
                    "role": role,
                    "content": [{"type": "text", "text": text}],
                }
                if extra_content:
                    new_msg["extra_content"] = extra_content
                messages.append(new_msg)
            elif role == "user":
                # User messages can have both images and text
                user_content = []
                for item in msg["content"]:
                    if item["type"] == "image" and "image" in item:
                        # Convert PIL Image to base64 data URL
                        img_b64 = self._image_to_base64(item["image"])
                        user_content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_b64}"
                                },
                            }
                        )
                    elif item["type"] == "text":
                        user_content.append(
                            {"type": "text", "text": item["text"]}
                        )

                messages.append({"role": "user", "content": user_content})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                reasoning_effort=self.reasoning_effort,
            )
            for msg in messages:
                if msg["role"] == "user":
                    for item in msg["content"]:
                        if item["type"] == "image_url":
                            item["image_url"]["url"] = "[IMAGE]"

            thought_signature = None
            usage = response.usage
            completion_tokens = usage.completion_tokens
            prompt_tokens = usage.prompt_tokens
            total_tokens = usage.total_tokens
            thinking_tokens = total_tokens - (completion_tokens + prompt_tokens)

            # Try to get the thought signature from the extra content
            msg_obj = response.choices[0].message
            extra = getattr(msg_obj, "extra_content", None)
            if isinstance(extra, dict):
                thought_signature = extra.get("google", {}).get(
                    "thought_signature", None
                )

            response_data = {
                "text": response.choices[0].message.content,
                "thinking_tokens": thinking_tokens,
                "completion_tokens": completion_tokens,
            }

            if thought_signature:
                response_data["extra_content"] = {
                    "google": {"thought_signature": thought_signature},
                }

            return response_data
        except Exception as e:
            raise RuntimeError(f"API Error: {e}")
