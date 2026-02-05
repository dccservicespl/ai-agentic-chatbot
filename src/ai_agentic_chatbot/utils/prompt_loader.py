from pathlib import Path


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


def get_system_prompt() -> str:
    import os
    import datetime

    now = datetime.datetime.now()
    formatted_date = now.strftime("%A, %B %d, %Y")
    prompt_text = load_file_content(os.environ["SYSTEM_PROMPT_PATH"])
    return prompt_text.format(formatted_date=formatted_date)
