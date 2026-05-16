import asyncio
import logging
import time
from typing import Any

from google import genai
from google.genai import types

from config import GOOGLE_API_KEY, NON_LIVE_LLM_TIMEOUT_SECONDS


NON_LIVE_LLM_TIMEOUT_MILLISECONDS = max(NON_LIVE_LLM_TIMEOUT_SECONDS, 1) * 1000

# Retryable HTTP error codes and keywords for exception text matching
_RETRYABLE_ERROR_CODES = ("408", "429", "500", "502", "503", "504", "DEADLINE_EXCEEDED")


def _is_retryable_error(exc: Exception) -> bool:
    """Determines whether to retry on this error."""
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    exc_str = str(exc)
    return any(code in exc_str for code in _RETRYABLE_ERROR_CODES)


def build_non_live_genai_client() -> genai.Client:
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY is not set in configs or .env")
    return genai.Client(
        api_key=GOOGLE_API_KEY,
        http_options=types.HttpOptions(timeout=NON_LIVE_LLM_TIMEOUT_MILLISECONDS),
    )


async def generate_content_with_diagnostics_async(
    *,
    client: genai.Client,
    model: str,
    contents: Any,
    config: types.GenerateContentConfig,
    logger: logging.Logger,
    operation_name: str,
    max_retries: int = 1,
):
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        started_at = time.monotonic()
        logger.info(
            "LLM async request started: op=%s model=%s attempt=%d/%d timeout_s=%s",
            operation_name,
            model,
            attempt + 1,
            max_retries + 1,
            NON_LIVE_LLM_TIMEOUT_SECONDS,
        )
        try:
            response = await client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except asyncio.CancelledError:
            logger.warning("LLM async request cancelled: op=%s", operation_name)
            raise
        except Exception as exc:
            duration_ms = (time.monotonic() - started_at) * 1000.0
            last_exc = exc
            if attempt < max_retries and _is_retryable_error(exc):
                wait_time = 2 ** attempt  # 1s, 2s, 4s...
                logger.warning(
                    "LLM async request retry: op=%s model=%s attempt=%d/%d duration_ms=%.1f wait=%.1fs error=%s",
                    operation_name,
                    model,
                    attempt + 1,
                    max_retries + 1,
                    duration_ms,
                    wait_time,
                    exc,
                )
                await asyncio.sleep(wait_time)
                continue
            logger.error(
                "LLM async request failed: op=%s model=%s attempt=%d/%d duration_ms=%.1f error=%s",
                operation_name,
                model,
                attempt + 1,
                max_retries + 1,
                duration_ms,
                exc,
            )
            raise
        duration_ms = (time.monotonic() - started_at) * 1000.0
        logger.info(
            "LLM async request finished: op=%s model=%s attempt=%d/%d duration_ms=%.1f",
            operation_name,
            model,
            attempt + 1,
            max_retries + 1,
            duration_ms,
        )
        return response
    raise last_exc

