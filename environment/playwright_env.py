import os
from playwright.async_api import async_playwright


class PlaywrightEnv:
    """Environment for interacting with HTML pages via Playwright."""

    def __init__(self, headless=True, viewport_width=1280, viewport_height=720):
        """Initialize the environment (browser setup happens in setup())."""
        self.headless = headless
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.playwright = None
        self.browser = None
        self.page = None

    async def setup(self):
        """Async setup of playwright, browser, and page."""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless
        )
        # Create initial page
        await self.reset()

    async def reset(self):
        """Reset the environment by closing the current page/context and creating a new one."""
        if self.page:
            await self.page.close()

        # browser.new_page() creates a new page in a new isolated context
        self.page = await self.browser.new_page(
            viewport={
                "width": self.viewport_width,
                "height": self.viewport_height,
            }
        )

    async def cleanup(self):
        """Close browser and cleanup resources."""
        if self.page:
            await self.page.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    async def load_html(self, html_file):
        """Load an HTML file into the page."""
        if not os.path.exists(html_file):
            raise FileNotFoundError(f"HTML file not found: {html_file}")

        file_url = f"file://{os.path.abspath(html_file)}"
        await self.page.goto(file_url)
        await self.page.wait_for_load_state("networkidle")

    async def click(self, x, y):
        """Click at the specified coordinates."""
        await self.page.mouse.click(x, y)

    async def shift_click(self, x, y):
        """Shift click at the specified coordinates."""
        await self.page.keyboard.down("Shift")
        await self.click(x, y)
        await self.page.keyboard.up("Shift")

    async def hover(self, x, y):
        """Hover at the specified coordinates."""
        await self.page.mouse.move(x, y)

    async def scroll(self, delta_x=0, delta_y=0):
        """Scroll by the specified deltas."""
        await self.page.mouse.wheel(delta_x, delta_y)

    async def _press_arrow_key(self, key, n=1):
        """Press the arrow key n times."""
        for _ in range(n):
            await self.page.keyboard.press(key)

    async def page_down(self, n=1):
        """Scroll down by n pages."""
        await self._press_arrow_key("PageDown", n)

    async def page_up(self, n=1):
        """Scroll up by n pages."""
        await self._press_arrow_key("PageUp", n)

    async def arrow_right(self, n=1):
        """Press the right arrow key n times."""
        await self._press_arrow_key("ArrowRight", n)

    async def arrow_left(self, n=1):
        """Press the left arrow key n times."""
        await self._press_arrow_key("ArrowLeft", n)

    async def arrow_up(self, n=1):
        """Press the up arrow key n times."""
        await self._press_arrow_key("ArrowUp", n)

    async def arrow_down(self, n=1):
        """Press the down arrow key n times."""
        await self._press_arrow_key("ArrowDown", n)

    async def drag(self, start_pos, end_pos):
        """Drag from start_pos to end_pos."""
        await self.hover(*start_pos)
        await self.page.mouse.down()
        await self.hover(*end_pos)
        await self.page.mouse.up()

    async def screenshot(self):
        """Take a screenshot and return the bytes."""
        return await self.page.screenshot()

    async def wait(self, timeout_ms=200):
        """Wait for a specified timeout in milliseconds."""
        await self.page.wait_for_timeout(timeout_ms)

    async def execute_action(self, action: dict) -> None:
        """
        Execute an action based on the action dictionary.

        Args:
            action: Dictionary with 'type' and action-specific parameters
                Examples:
                - {"type": "click", "x": 100, "y": 200}
                - {"type": "scroll", "dx": 0, "dy": 100}
                - {"type": "drag", "start_pos": (10, 20), "end_pos": (100, 200)}
                - {"type": "answer", "value": "42"}

        Raises:
            ValueError: If action type is unknown or answer type (not executable)
        """
        action_type = action["type"]

        if action_type == "click":
            await self.click(action["x"], action["y"])
        elif action_type == "shift_click":
            await self.shift_click(action["x"], action["y"])
        elif action_type == "hover":
            await self.hover(action["x"], action["y"])
        elif action_type == "scroll":
            await self.scroll(action["dx"], action["dy"])
        elif action_type == "page_down":
            await self.page_down(action["n"])
        elif action_type == "page_up":
            await self.page_up(action["n"])
        elif action_type == "arrow_right":
            await self.arrow_right(action["n"])
        elif action_type == "arrow_left":
            await self.arrow_left(action["n"])
        elif action_type == "arrow_up":
            await self.arrow_up(action["n"])
        elif action_type == "arrow_down":
            await self.arrow_down(action["n"])
        elif action_type == "drag":
            await self.drag(action["start_pos"], action["end_pos"])
        elif action_type == "answer":
            # Answer is not an executable action
            raise ValueError(f"Action type 'answer' is not executable")
        else:
            raise ValueError(f"Unknown action type: {action_type}")

    def get_action_description(self, action: dict) -> str:
        """
        Get a human-readable description of an action.

        Args:
            action: Action dictionary

        Returns:
            String description of the action
        """
        action_type = action["type"]

        if action_type in ["click", "shift_click", "hover"]:
            return f"{action_type} at ({action['x']}, {action['y']})"
        elif action_type == "scroll":
            return f"scroll by ({action['dx']}, {action['dy']})"
        elif action_type in [
            "page_down",
            "page_up",
            "arrow_right",
            "arrow_left",
            "arrow_up",
            "arrow_down",
        ]:
            return f"{action_type}({action['n']})"
        elif action_type == "drag":
            start = action["start_pos"]
            end = action["end_pos"]
            return f"drag from ({start[0]}, {start[1]}) to ({end[0]}, {end[1]})"
        elif action_type == "answer":
            return f"answer: {action['value']}"
        else:
            return f"unknown action: {action_type}"
