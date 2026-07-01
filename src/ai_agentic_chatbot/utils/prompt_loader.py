from pathlib import Path
from typing import Optional


def load_file_content(file_path: str | Path) -> str:
    """
    Load and return the content of a text-based file (e.g., .md, .txt).

    Args:
        file_path: Path to the file

    Returns:
        File content as a string

    Raises:
        FileNotFoundError: If the file does not exist
        IsADirectoryError: If the path points to a directory
        ValueError: If the file is empty
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    if not path.is_file():
        raise IsADirectoryError(f"Expected a file, got a directory: {path.resolve()}")

    content = path.read_text(encoding="utf-8").strip()

    if not content:
        raise ValueError(f"File is empty: {path.resolve()}")

    return content


def get_system_prompt(system_prompt_path: Optional[str] = None) -> str:
    """Load the system prompt, injecting today's date.

    system_prompt_path: explicit path (e.g. DbContextConfig.system_prompt_path
    for a given context). Falls back to the SYSTEM_PROMPT_PATH env var when
    None, preserving single-context behavior until callers are made
    context-aware (TODO 12/14).
    """
    import os
    import datetime

    now = datetime.datetime.now()
    formatted_date = now.strftime("%A, %B %d, %Y")
    path = system_prompt_path or os.environ["SYSTEM_PROMPT_PATH"]
    prompt_text = load_file_content(path)
    return prompt_text.format(formatted_date=formatted_date)


def get_router_prompt(router_prompt_path: Optional[str] = None) -> str:
    """Load the router prompt template (raw — caller formats in schema_text/etc.).

    router_prompt_path: explicit path (e.g. DbContextConfig.router_prompt_path
    for a given context). Falls back to the ROUTER_PROMPT_PATH env var when
    None, preserving single-context behavior until callers are made
    context-aware (TODO 12).
    """
    import os

    path = router_prompt_path or os.environ["ROUTER_PROMPT_PATH"]
    return load_file_content(path)
