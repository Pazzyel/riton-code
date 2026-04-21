# A SYSTEM PROMPT INCLUDING:
# - core
# - tools
# - skills
# - memory
# - agents_md
# - dynamic_context

from typing import List, Tuple
from pathlib import Path
import aiofiles
import aiofiles.os
import datetime
import os

from directory import WORKDIR, USER_AGENTS_MD, PROJECT_AGENTS_MD
from skill import SKILL_REGISTRY
from memory.memory import memory_manager

class SystemPromptBuilder:
    def __init__(self, work_dir: Path = WORKDIR):
        self.work_dir = work_dir
        self.skill_dir: Path = work_dir / "skills"
        self.memory_dir: Path = work_dir / ".memory"

    def _build_core(self) -> str:
        return f"You are a coding agent at {str(self.work_dir)}. \n" \
                "Use provided tools to explore, read, write, and modify files in the workspace. \n" \
                "Always verify your first. Prefer getting information by tools instead of making assumptions. \n" \

    # Tool information is not necessary. Because OpenAI SDK client will bind tools provided and offer tool prompts

    def _build_skills(self) -> str:
        return  "<skills>\n" \
                f"{SKILL_REGISTRY.describe_available()}" \
                "</skills>\n"

    def _build_memory(self) -> str:
        # For now, we just dump all memories into the system prompt. In the future, we can use a more sophisticated way to select relevant memories.
        return "<memories>\n" \
               f"{memory_manager.build_memory_prompt()}" \
               "</memories>\n"

    async def _build_agents_md(self) -> str:
        sources: List[Tuple[str, str]] = []
        if await aiofiles.os.path.exists(USER_AGENTS_MD):
             async with aiofiles.open(USER_AGENTS_MD, "r") as f:
                sources.append(("USER GLOBAL AGENTS.md in ~/.riton/AGENTS.md", await f.read()))
        if await aiofiles.os.path.exists(PROJECT_AGENTS_MD):
             async with aiofiles.open(PROJECT_AGENTS_MD, "r") as f:
                sources.append(("PROJECT ROOT AGENTS.md in ./AGENTS.md", await f.read()))
        cwd: Path = Path.cwd()
        if cwd != WORKDIR:
            # If the current working directory is different from the agent's work directory, we can also look for AGENTS.md in the current working directory
            cwd_agents_md: Path = cwd / "AGENTS.md"
            if await aiofiles.os.path.exists(cwd_agents_md):
                async with aiofiles.open(cwd_agents_md, "r") as f:
                    sources.append((f"SUBDIRECTORY AGENTS.md in {str(cwd)}/AGENTS.md", await f.read()))

        if not sources:
            return "<agents_md>\nNo AGENTS.md found.\n</agents_md>\n"
        
        parts: List[str] = []
        for label, content in sources:
            parts.append(f"## From {label}")
            parts.append(content.strip())
        return "<agents_md>\n" + "\n\n".join(parts) + "\n</agents_md>\n"

    def _build_dynamic_context(self) -> str:
        """Date, working directory, system..."""
        lines: List[str] = [
            f"Current date and time: {datetime.datetime.now().isoformat()}",
            f"Current working directory: {str(WORKDIR)}",
            f"Platform: {os.uname().sysname}",
        ]
        return "<dynamic_context>\n" + "\n".join(lines) + "\n</dynamic_context>\n"
    
    async def build(self) -> str:
        """Build the full system prompt by combining core, skills, memory, agents_md, and dynamic context."""
        core: str = self._build_core()
        skills: str = self._build_skills()
        memory: str = self._build_memory()
        agents_md: str = await self._build_agents_md()
        dynamic_context: str = self._build_dynamic_context()

        return "\n".join([core, skills, memory, agents_md, dynamic_context])
    
system_prompt_builder: SystemPromptBuilder = SystemPromptBuilder()