import logging

from google.genai import types

from config import GOOGLE_API_KEY
from memory_engine.llm_client_utils import build_non_live_genai_client, generate_content_with_diagnostics_async
from memory_engine.memory_config import SUMMARIZE_CONTEXT_GEMINI_MODEL, SUMMARIZE_CONTEXT_GEMINI_THINKING_LEVEL


logger = logging.getLogger(__name__)


class ContextAgent:
    def __init__(self) -> None:
        if not GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY is not set in configs or .env")
        self.client = build_non_live_genai_client()

    async def summarize_day_async(self, day_markdown: str) -> str:
        prompt = (
            "Provide a brief summary of the entire day's context in English. "
            "Make a bullet-point digest. Write as concisely as possible, in telegraphic style. No filler words. Example: '- Agreed on a surprise for Mikky's birthday. - Planned a search for a garage spot.'"
            "Return 1-2 paragraphs without lists. "
            "Preserve facts, stable preferences, agreements, plans."
            "Do not invent or assume extra details.\n\n"
            "Source daily log:\n"
            f"{day_markdown}"
        )

        response = await generate_content_with_diagnostics_async(
            client=self.client,
            model=SUMMARIZE_CONTEXT_GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_level=SUMMARIZE_CONTEXT_GEMINI_THINKING_LEVEL
                )
            ),
            logger=logger,
            operation_name="summarize_day",
        )
        return response.text.strip() if response.text else ""
