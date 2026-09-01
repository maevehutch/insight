from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, Dict


class BaseVLM(ABC):
    """Abstract base class for VLMs."""

    @abstractmethod
    def query(
        self,
        conversation_history: List[Dict],
    ) -> dict:
        """
        Query the VLM with a complete conversation history.

        Args:
            conversation_history: Complete conversation history in multimodal format.
                Should include system message at the start, followed by user/assistant pairs.
                The last message should be a user message with the current screenshot and prompt.
                Format: [
                    {"role": "system", "content": [{"type": "text", "text": "..."}]},
                    {"role": "user", "content": [
                        {"type": "image", "image": <PIL.Image>},
                        {"type": "text", "text": "..."}
                    ]},
                    {"role": "assistant", "content": [{"type": "text", "text": "..."}]},
                    {"role": "user", "content": [...]},  # Current turn
                ]

        Returns:
            dict: The VLM's response text and other metadata
        """
        pass

    def extract_action(self, response: str) -> Optional[dict]:
        """
        Extract action from VLM response.

        Expected formats inside <action> tags:
        - click(x, y)
        - shift_click(x, y)
        - hover(x, y)
        - scroll(dx, dy)
        - page_down(n)
        - page_up(n)
        - arrow_right(n)
        - arrow_left(n)
        - arrow_up(n)
        - arrow_down(n)
        - drag((x1, y1), (x2, y2))
        - answer(True | False | Not Enough Information)

        Args:
            response: VLM response text

        Returns:
            dict with action info or None if no valid action found
        """
        import re

        # Find all <action> tags
        # Using DOTALL to handle potential newlines
        action_matches = re.findall(
            r"<action>(.*?)</action>", response, re.DOTALL | re.IGNORECASE
        )

        if not action_matches:
            return None

        # Take the last match to handle reasoning/chain-of-thought
        command = action_matches[-1].strip()
        command_lower = command.lower()

        # Check for answer
        # Format: answer(True), answer(False), answer(Not Enough Information)
        answer_match = re.search(r"answer\s*\((.*?)\)", command, re.IGNORECASE)
        if answer_match:
            val = answer_match.group(1).strip()
            return {"type": "answer", "value": val}

        # Check for drag (must come before click to avoid mismatching)
        drag_match = re.search(
            r"drag\s*\(\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*\)",
            command_lower,
        )
        if drag_match:
            x1, y1 = int(drag_match.group(1)), int(drag_match.group(2))
            x2, y2 = int(drag_match.group(3)), int(drag_match.group(4))
            return {"type": "drag", "start_pos": (x1, y1), "end_pos": (x2, y2)}

        # Check for shift_click (must come before click to avoid mismatching)
        shift_click_match = re.search(
            r"shift_click\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", command_lower
        )
        if shift_click_match:
            x, y = int(shift_click_match.group(1)), int(
                shift_click_match.group(2)
            )
            return {"type": "shift_click", "x": x, "y": y}

        # Check for click
        click_match = re.search(
            r"click\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", command_lower
        )
        if click_match:
            x, y = int(click_match.group(1)), int(click_match.group(2))
            return {"type": "click", "x": x, "y": y}

        # Check for hover
        hover_match = re.search(
            r"hover\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", command_lower
        )
        if hover_match:
            x, y = int(hover_match.group(1)), int(hover_match.group(2))
            return {"type": "hover", "x": x, "y": y}

        # Check for scroll
        scroll_match = re.search(
            r"scroll\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", command_lower
        )
        if scroll_match:
            dx, dy = int(scroll_match.group(1)), int(scroll_match.group(2))
            return {"type": "scroll", "dx": dx, "dy": dy}

        # Check for page_down
        page_down_match = re.search(
            r"page_down\s*\(\s*(\d+)\s*\)", command_lower
        )
        if page_down_match:
            n = int(page_down_match.group(1))
            return {"type": "page_down", "n": n}

        # Check for page_up
        page_up_match = re.search(r"page_up\s*\(\s*(\d+)\s*\)", command_lower)
        if page_up_match:
            n = int(page_up_match.group(1))
            return {"type": "page_up", "n": n}

        # Check for arrow_right
        arrow_right_match = re.search(
            r"arrow_right\s*\(\s*(\d+)\s*\)", command_lower
        )
        if arrow_right_match:
            n = int(arrow_right_match.group(1))
            return {"type": "arrow_right", "n": n}

        # Check for arrow_left
        arrow_left_match = re.search(
            r"arrow_left\s*\(\s*(\d+)\s*\)", command_lower
        )
        if arrow_left_match:
            n = int(arrow_left_match.group(1))
            return {"type": "arrow_left", "n": n}

        # Check for arrow_up
        arrow_up_match = re.search(r"arrow_up\s*\(\s*(\d+)\s*\)", command_lower)
        if arrow_up_match:
            n = int(arrow_up_match.group(1))
            return {"type": "arrow_up", "n": n}

        # Check for arrow_down
        arrow_down_match = re.search(
            r"arrow_down\s*\(\s*(\d+)\s*\)", command_lower
        )
        if arrow_down_match:
            n = int(arrow_down_match.group(1))
            return {"type": "arrow_down", "n": n}

        return None

    def extract_action_ui_tars(self, response: str) -> Optional[dict]:
        """
        Extract UI-TARS-style action from VLM response.

        Expected response format (see `prompts/ui_tars_user_first_interaction.txt`):
          Thought: ...
          Action: ...

        Parsing rule:
        - Find the LAST occurrence of "Action:" in the response.
        - Treat everything AFTER it as the action string (strip whitespace).
        - Parse actions using the UI-TARS action syntax in `ui_tars_user_first_interaction.txt`.

        Supported action syntax:
        - click(start_box='(x, y)')
        - shift_click(start_box='(x, y)')
        - hover(start_box='(x, y)')
        - scroll(dx, dy)
        - page_down(n), page_up(n)
        - arrow_right(n), arrow_left(n), arrow_up(n), arrow_down(n)
        - drag(start_box='(x1, y1)', end_box='(x2, y2)')
        - answer(True | False | Not Enough Information)
        """
        import re

        # Find last "Action:" and take the rest of the completion
        matches = list(re.finditer(r"\bAction\s*:\s*", response, re.IGNORECASE))
        if not matches:
            return None

        last = matches[-1]
        action_text = response[last.end() :].strip()
        if not action_text:
            return None

        # If the model wraps the action in a code fence, strip it.
        action_text = action_text.strip()
        if action_text.startswith("```"):
            # Drop the opening fence line
            parts = action_text.split("\n", 1)
            action_text = parts[1].strip() if len(parts) == 2 else ""
            # Drop trailing fence
            if "```" in action_text:
                action_text = action_text.split("```", 1)[0].strip()
        if not action_text:
            return None

        # Normalize to first non-empty line for robust parsing
        first_line = next(
            (ln.strip() for ln in action_text.splitlines() if ln.strip()), ""
        )
        if not first_line:
            return None

        cmd = first_line

        # answer(...)
        m = re.search(r"answer\s*\(\s*(.*?)\s*\)", cmd, re.IGNORECASE)
        if m:
            return {"type": "answer", "value": m.group(1).strip()}

        # drag(start_box='(x1, y1)', end_box='(x2, y2)')
        m = re.search(
            r"drag\s*\(\s*start_box\s*=\s*['\"]\(\s*(\d+)\s*,\s*(\d+)\s*\)['\"]\s*,\s*end_box\s*=\s*['\"]\(\s*(\d+)\s*,\s*(\d+)\s*\)['\"]\s*\)",
            cmd,
            re.IGNORECASE,
        )
        if m:
            x1, y1, x2, y2 = (int(m.group(i)) for i in range(1, 5))
            return {"type": "drag", "start_pos": (x1, y1), "end_pos": (x2, y2)}

        # click/shift_click/hover(start_box='(x, y)')
        for name in ("shift_click", "click", "hover"):
            m = re.search(
                rf"{name}\s*\(\s*start_box\s*=\s*['\"]\(\s*(\d+)\s*,\s*(\d+)\s*\)['\"]\s*\)",
                cmd,
                re.IGNORECASE,
            )
            if m:
                x, y = int(m.group(1)), int(m.group(2))
                if name == "shift_click":
                    return {"type": "shift_click", "x": x, "y": y}
                if name == "click":
                    return {"type": "click", "x": x, "y": y}
                return {"type": "hover", "x": x, "y": y}

        # scroll(dx, dy)
        m = re.search(
            r"scroll\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", cmd, re.IGNORECASE
        )
        if m:
            return {
                "type": "scroll",
                "dx": int(m.group(1)),
                "dy": int(m.group(2)),
            }

        # page_down/page_up/arrows
        for name in (
            "page_down",
            "page_up",
            "arrow_right",
            "arrow_left",
            "arrow_up",
            "arrow_down",
        ):
            m = re.search(rf"{name}\s*\(\s*(\d+)\s*\)", cmd, re.IGNORECASE)
            if m:
                return {"type": name, "n": int(m.group(1))}

        return None
