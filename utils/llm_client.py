"""
LLM Client — Google Gemini API.
"""

import os
import json
import time
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

MODEL          = "gemini-2.5-flash"
RETRY_ATTEMPTS = 4
RETRY_DELAY    = 15


def call_claude(
    system_prompt: str,
    user_message: str,
    expect_json: bool = False,
    max_tokens: int = 8192,
) -> str | dict | list:

    model = genai.GenerativeModel(
        model_name=MODEL,
        system_instruction=system_prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.0,
            max_output_tokens=max_tokens,
        ),
    )

    last_error = None

    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = model.generate_content(user_message)
            raw_text = response.text.strip()

            if not expect_json:
                return raw_text

            cleaned = raw_text
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]
                cleaned = cleaned.rsplit("```", 1)[0].strip()

            return json.loads(cleaned)

        except json.JSONDecodeError as e:
            partial = _try_recover_json(cleaned if "cleaned" in dir() else "")
            if partial is not None:
                return partial
            raise e

        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            if "429" in error_str or "quota" in error_str or "rate" in error_str:
                delay = RETRY_DELAY * (attempt + 1)
                print(f"[LLMClient] Rate limit hit. Waiting {delay}s "
                      f"(attempt {attempt + 1}/{RETRY_ATTEMPTS})...")
                time.sleep(delay)
                continue
            raise e

    raise last_error


def _try_recover_json(text: str) -> list | dict | None:
    if not text.strip().startswith("["):
        return None
    try:
        last_close = text.rfind("}")
        if last_close == -1:
            return None
        truncated = text[:last_close + 1] + "]"
        return json.loads(truncated)
    except Exception:
        return None
