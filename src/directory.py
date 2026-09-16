from pathlib import Path

WORKDIR: Path = Path.cwd()
CONFIG_DIR: Path = WORKDIR / ".riton"
SKILL_DIR: Path = CONFIG_DIR / "skills"
TRANSCRIPT_DIR: Path = CONFIG_DIR / ".transcripts"
TOOL_RESULTS_DIR: Path = CONFIG_DIR / ".task_outputs" / "tool-results"
MEMORY_DIR: Path = CONFIG_DIR / ".memory"
MEMORY_INDEX: Path = MEMORY_DIR / "MEMORY.md"
USER_AGENTS_MD: Path = Path.home() / ".riton" / "AGENTS.md"
PROJECT_AGENTS_MD: Path = CONFIG_DIR / "AGENTS.md"
TASKS_DIR: Path = CONFIG_DIR / ".tasks"
# RUNTIME_DIR: Path = CONFIG_DIR / ".runtime-tasks"
DURABLE_PATH = CONFIG_DIR / ".scheduled_tasks.json"