"""
LLM Client — DeepSeek API
"""

import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

MODEL = os.getenv("MODEL_NAME")

RETRY_ATTEMPTS = 4
RETRY_DELAY = 60


def call_claude(
    system_prompt: str,
    user_message: str,
    expect_json: bool = False,
    max_tokens: int = 8192,
) -> str | dict | list:

    last_error = None

    for attempt in range(RETRY_ATTEMPTS):

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_message,
                    },
                ],
                temperature=0.0,
                max_tokens=max_tokens,
            )

            raw_text = response.choices[0].message.content.strip()

            if not expect_json:
                return raw_text

            cleaned = raw_text

            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]
                cleaned = cleaned.rsplit("```", 1)[0].strip()

            return json.loads(cleaned)

        except json.JSONDecodeError as e:

            partial = _try_recover_json(
                cleaned if "cleaned" in locals() else ""
            )

            if partial is not None:
                return partial

            raise e

        except Exception as e:

            last_error = e
            error_str = str(e).lower()

            if (
                "429" in error_str
                or "quota" in error_str
                or "rate" in error_str
            ):
                delay = RETRY_DELAY * (attempt + 1)

                print(
                    f"[LLMClient] Rate limit hit. "
                    f"Waiting {delay}s "
                    f"(attempt {attempt + 1}/{RETRY_ATTEMPTS})..."
                )

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

        truncated = text[: last_close + 1] + "]"

        return json.loads(truncated)

    except Exception:
        return None
