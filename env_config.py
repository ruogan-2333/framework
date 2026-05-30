r"""Project environment loading helpers.

Input:
  - A project .env path, usually F:\workplace\framework\.env.

Output:
  - The current Python process environment is updated from that .env file.

Function:
  - Make API and proxy settings deterministic by clearing known inherited
    variables before loading the project .env file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv


PROJECT_ENV_KEYS: tuple[str, ...] = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
)


def load_project_env(env_path: Path | None = None, *, clear_keys: Iterable[str] = PROJECT_ENV_KEYS) -> bool:
    """Load the project .env file after clearing inherited API/proxy variables.

    Input:
      - env_path: optional path to the .env file. When omitted, the .env next to
        this helper file is used.
      - clear_keys: environment variable names that should not be inherited from
        the parent shell.

    Output:
      - True when python-dotenv found and loaded the .env file, otherwise False.

    Function:
      - Ensure the project .env values are the source of truth for API keys,
        API base URL, and proxy settings during this process and child processes.
    """

    resolved_env_path = env_path or Path(__file__).resolve().parent / ".env"
    for key in clear_keys:
        os.environ.pop(key, None)
    return bool(load_dotenv(resolved_env_path, override=True))
