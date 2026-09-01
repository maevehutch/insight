import os
from typing import Any, List, Dict, Optional
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

from .base import BaseVLM


class HuggingFaceModel(BaseVLM):
    """Hugging Face Vision Language Model implementation."""

    def __init__(
        self,
        model_path: str = None,
        prompts_dir: str = None,
        prompt_prefix: Optional[str] = None,
        # Generation params
        do_sample: bool = True,
        top_p: float = 0.95,
        top_k: int = 20,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        default_context_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ):
        """
        Initialize Hugging Face model.

        Args:
            model_path: Path to the model
            prompts_dir: Path to the directory containing prompt files.
                        If None, defaults to 'prompts' directory relative to this file.
        """
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.torch = torch

        # GPU sanity check
        assert torch.cuda.is_available(), "CUDA is not available"

        # For device_map="auto", model.device can be misleading — check parameters
        assert next(
            self.model.parameters()
        ).is_cuda, "Model parameters are not on CUDA (CPU fallback occurred)"

        print(f"HF device map: {self.model.hf_device_map}")
        print(f"Model device: {self.model.device}")

        # Store generation params
        self.do_sample = bool(do_sample)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.repetition_penalty = float(repetition_penalty)
        self.temperature = float(temperature)
        self.max_new_tokens = (
            max_new_tokens if max_new_tokens is None else int(max_new_tokens)
        )

        # Context length: use provided default if set, else infer from config/tokenizer
        self.context_length = (
            int(default_context_length) if default_context_length else 0
        )
        if self.context_length <= 0:
            if hasattr(self.model.config, "max_position_embeddings"):
                self.context_length = int(
                    self.model.config.max_position_embeddings
                )
            elif hasattr(self.model.config, "text_config") and hasattr(
                self.model.config.text_config, "max_position_embeddings"
            ):
                self.context_length = int(
                    self.model.config.text_config.max_position_embeddings
                )
            else:
                pass

        if self.context_length <= 0 and self.max_new_tokens is None:
            raise ValueError(
                "Could not infer context length from model config/tokenizer. "
                "Please pass --default-context-length or --max-new-tokens."
            )

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

    def query(
        self,
        conversation_history: List[Dict],
    ) -> dict:
        """
        Query Hugging Face model with complete conversation history.

        Args:
            conversation_history: Complete conversation with current turn's user message.
                Format: [{"role": "...", "content": [{"type": "text"|"image", ...}]}]

        Returns:
            Model response text
        """
        # Build messages list for chat template and extract images
        messages = []
        images = []

        # Process conversation history: extract images and build clean messages
        for msg in conversation_history:
            # Build a clean message for the chat template (without actual image objects)
            clean_msg = {"role": msg["role"], "content": []}

            for item in msg["content"]:
                if item["type"] == "image":
                    # Extract the PIL Image for the images list
                    if "image" in item:
                        images.append(item["image"])
                    # Add placeholder for chat template (without the image object)
                    clean_msg["content"].append({"type": "image"})
                elif item["type"] == "text":
                    # Copy text content as-is
                    clean_msg["content"].append(
                        {"type": "text", "text": item["text"]}
                    )

            messages.append(clean_msg)

        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

        # Process with ALL images - the processor handles model-specific image tokens
        inputs = self.processor(
            text=[prompt],
            images=images,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        input_len = int(inputs.input_ids.shape[-1])
        if self.max_new_tokens is not None:
            max_new_tokens = int(self.max_new_tokens)
        else:
            # Use context window to determine how many new tokens to generate
            # Clamp to at least 1 token to avoid generation errors
            max_new_tokens = max(self.context_length - input_len, 1)

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }

        with self.torch.no_grad():
            generated_ids = self.model.generate(
                pad_token_id=self.processor.tokenizer.pad_token_id,
                **inputs,
                **gen_kwargs,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # Decode output
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # Token counting
        gen_ids = generated_ids_trimmed[0].tolist()
        thinking_tokens = 0
        total_tokens = len(gen_ids)

        try:
            # UI-TARS "Thought:/Action:" token counting:
            if "Action:" in response and "</think>" not in response:
                import re

                action_matches = list(
                    re.finditer(r"\bAction\s*:\s*", response, re.IGNORECASE)
                )
                if action_matches:
                    action_m = action_matches[-1]
                    action_body = response[action_m.end() :].strip()
                    completion_tokens = len(
                        self.processor.tokenizer.encode(
                            action_body, add_special_tokens=False
                        )
                    )
                    thinking_tokens = max(total_tokens - completion_tokens, 0)
                else:
                    completion_tokens = total_tokens
            elif "</think>" in response:
                try:
                    parts = response.split("</think>")
                    if len(parts) >= 2:
                        # 1. Try single token match for </think>
                        separator = "</think>"
                        separator_ids = self.processor.tokenizer.encode(
                            separator, add_special_tokens=False
                        )

                        found_split = False
                        if len(separator_ids) == 1:
                            sep_id = separator_ids[0]
                            if sep_id in gen_ids:
                                thinking_tokens = gen_ids.index(sep_id)
                                found_split = True

                        # 2. Fallback: tokenize decoded reasoning text
                        if not found_split:
                            thinking_ids = self.processor.tokenizer.encode(
                                parts[0] + separator, add_special_tokens=False
                            )
                            thinking_tokens = len(thinking_ids)
                except Exception:
                    pass

                completion_tokens = max(total_tokens - thinking_tokens, 0)
            else:
                completion_tokens = total_tokens
        except Exception:
            completion_tokens = max(total_tokens - thinking_tokens, 0)

        return {
            "text": response,
            "thinking_tokens": int(thinking_tokens),
            "completion_tokens": int(completion_tokens),
        }
