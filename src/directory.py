from pathlib import Path

WORKDIR: Path = Path.cwd()
SKILL_DIR: Path = WORKDIR / "skills"
TRANSCRIPT_DIR: Path = WORKDIR / ".transcripts"
TOOL_RESULTS_DIR: Path = WORKDIR / ".task_outputs" / "tool-results"