"""Settings loaded from the environment (.env in development).

Every module imports settings from here instead of calling os.environ
directly, so a missing key fails in one place with a clear message.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    github_token: str | None
    groq_api_key: str | None


def get_settings() -> Settings:
    return Settings(
        github_token=os.environ.get("GITHUB_TOKEN") or None,
        groq_api_key=os.environ.get("GROQ_API_KEY") or None,
    )


def require(value: str | None, name: str) -> str:
    """Return the value, or stop with a message saying exactly what to fix."""
    if not value:
        raise RuntimeError(f"{name} is not set. Add it to your .env file (see .env.example).")
    return value
