from __future__ import annotations

from pathlib import Path
from typing import Iterable

from codebase_rag.domain.code_chunk import CodeChunk

DEFAULT_CHUNK_SIZE = 120
DEFAULT_OVERLAP = 20
SUPPORTED_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".cpp": "cpp",
    ".c": "c",
    ".cs": "csharp",
    ".html": "html",
    ".css": "css",
    ".json": "json",
    ".md": "markdown",
    ".txt": "text",
}

EXCLUDED_DIR_NAMES = {"__pycache__", ".git", "node_modules", ".venv", "venv"}


def _chunk_step(chunk_size: int, overlap: int) -> int:
    return max(chunk_size - overlap, 1)


def _language_for_path(path: Path) -> str:
    return SUPPORTED_EXTENSIONS.get(path.suffix.lower(), "text")


def scan_repository(
    repo_root: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    file_extensions: Iterable[str] | None = None,
) -> list[CodeChunk]:
    """Walk a repository and split source files into indexable code chunks.

    This function is intentionally simple so users can follow the full
    embedding pipeline from raw repository files to dense vectors.
    """
    repo_root = Path(repo_root).expanduser().resolve()
    if not repo_root.exists() or not repo_root.is_dir():
        raise ValueError(f"Repository path not found: {repo_root}")

    allowed_exts = {
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in (file_extensions or SUPPORTED_EXTENSIONS.keys())
    }

    chunks: list[CodeChunk] = []
    for path in sorted(repo_root.rglob("*")):
        if not path.is_file():
            continue

        if any(part in EXCLUDED_DIR_NAMES for part in path.parts):
            continue

        if path.suffix.lower() not in allowed_exts:
            continue

        text = path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()
        if not lines:
            continue

        step = _chunk_step(chunk_size, overlap)
        repo_name = repo_root.name
        relative_path = str(path.relative_to(repo_root)).replace("\\", "/")

        for start_index in range(0, len(lines), step):
            end_index = min(start_index + chunk_size, len(lines))
            snippet = "\n".join(lines[start_index:end_index]).strip()
            if not snippet:
                continue

            chunks.append(
                CodeChunk(
                    chunk_id=f"{relative_path}:{start_index + 1}-{end_index}",
                    file_path=relative_path,
                    language=_language_for_path(path),
                    start_line=start_index + 1,
                    end_line=end_index,
                    text=snippet,
                    repo_name=repo_name,
                )
            )

            if end_index == len(lines):
                break

    return chunks
