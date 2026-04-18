from typing import Dict, Any, Optional, Tuple
from pydantic import BaseModel
from pathlib import Path
import re
from re import Match
import yaml
import logging

# The file didn's use asyncio or aiofiles before
# Because it's io only happens in program initialization

from directory import SKILL_DIR

logger = logging.getLogger(__name__)

class SkillManifest(BaseModel):
    name: str
    description: str

class SkillDocument(BaseModel):
    manifest: SkillManifest
    body: str # Full markdown content of the SKILL.md file

class SkillRegistry:
    def __init__(self, skill_dir: Path = SKILL_DIR) -> None:
        self.skills: Dict[str, SkillDocument] = {}
        self.skill_dir: Path = skill_dir
        self._load_all()
    
    def _load_all(self) -> None:
        """Load all SKILL.md files from the skill directory and register them."""
        if not self.skill_dir.exists():
            logger.info(f"Skill directory {self.skill_dir} does not exist. No skills loaded.")
            return
        for skill_file in self.skill_dir.rglob("**/SKILL.md"):
            with skill_file.open() as f:
                content: str = f.read()
                meta, body = self._parse_frontmatter(content)
                if "name" in meta:
                    skill: SkillDocument = SkillDocument(
                        manifest=SkillManifest(name=meta["name"], description=meta.get("description", "No description provided.")),
                        body=body
                    )
                    self.register(skill)
                    logger.info(f"Loaded skill '{meta['name']}' from {skill_file}")
                    if "description" not in meta:
                        logger.warning(f"SKILL.md file at {skill_file} is missing a 'description' in its frontmatter. Consider adding one for better clarity.")
                else:
                    logger.warning(f"SKILL.md file at {skill_file} is missing a 'name' in its frontmatter and will be skipped.")

    def _parse_frontmatter(self, text: str) -> Tuple[Dict[str, Any], str]:
        """
        Parse YAML frontmatter between --- delimiters.
        
        Args:
            text: The input markdown SKILL.md text
        Returns:
            A tuple of (metadata dict, main content string)
        """
        # Match the markdown frontmatter format: ---\n<yaml frontmatter>\n---\n<rest of content>
        # It's a yaml frontmatter block that starts and ends with --- on its own line, and captures the content in between as YAML, and the rest of the text after the second --- as the main content.
        match: Optional[Match] = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
        if not match:
            return {}, text
        try:
            meta: Dict[str, Any] = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            meta: Dict[str, Any] = {}
        return meta, match.group(2).strip()

    def register(self, skill: SkillDocument) -> None:
        """Register a skill in the registry."""
        self.skills[skill.manifest.name] = skill

    def describe_available(self) -> str:
        """Return a string describing the available skills in the registry."""
        return "\n".join(f"- {name}: {skill.manifest.description}" for name, skill in self.skills.items())

    def get_content(self, name: str) -> str:
        """Get the full markdown content of a skill by name."""
        logger.info("Use skill: %s", name)
        skill = self.skills.get(name)
        return f"""
        <skill name="{name}">
        {skill.body if skill else 'Skill content not provided.'}
        </skill>
        """
    
    def get_skill_dir(self, name: str) -> str:
        """Get the directory path of a specific skill by name."""
        skill = self.skills.get(name)
        if not skill:
            raise ValueError(f"Skill '{name}' not found in registry.")
        # Assuming the skill's body contains a reference to its file path, we can extract it. 
        # For simplicity, let's assume the skill's body starts with a line like "Path: <relative_path_to_skill_file>"
        return (SKILL_DIR / name).as_posix()

SKILL_REGISTRY = SkillRegistry()

def load_skill(name: str) -> str:
    """Load a skill by name from the registry."""
    return SKILL_REGISTRY.get_content(name)

def get_skill_dir(name: str) -> str:
    """Get the directory path of a specific skill by name."""
    return SKILL_REGISTRY.get_skill_dir(name)