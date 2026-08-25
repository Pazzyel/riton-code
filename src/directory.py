from pathlib import Path

WORKDIR: Path = Path.cwd()
SKILL_DIR: Path = WORKDIR / "skills"
TRANSCRIPT_DIR: Path = WORKDIR / ".transcripts"
TOOL_RESULTS_DIR: Path = WORKDIR / ".task_outputs" / "tool-results"
MEMORY_DIR: Path = WORKDIR / ".memory"
MEMORY_INDEX: Path = MEMORY_DIR / "MEMORY.md"
USER_AGENTS_MD: Path = Path.home() / ".riton" / "AGENTS.md"
PROJECT_AGENTS_MD: Path = WORKDIR / "AGENTS.md"
TASKS_DIR: Path = WORKDIR / ".tasks"
# RUNTIME_DIR: Path = WORKDIR / ".runtime-tasks"