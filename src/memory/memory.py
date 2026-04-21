from typing_extensions import Literal, TypeAlias
from pydantic import BaseModel
from pathlib import Path
from typing import Dict, Optional, List
import re
import logging

import aiofiles
import aiofiles.os

from directory import MEMORY_DIR, WORKDIR, MEMORY_INDEX

logger = logging.getLogger(__name__)

MemoryType: TypeAlias = Literal["user", "feedback", "project", "reference"]
MAX_INDEX_LINES: int = 200 # Max lines to show in the index of MEMORY.md

class Memory(BaseModel):
    description: str
    type: MemoryType
    content: str
    file_name: str

# Format in MEMORY.md:
# - {name}: {description} [{type}]           # Each line is a memory
# ... (truncated at {MAX_INDEX_LINES} lines) # if more than {MAX_INDEX_LINES} lines

class MemoryManager:
    def __init__(self, memory_dir: Optional[Path] = None) -> None:
        self.memory_dir: Path = memory_dir or MEMORY_DIR
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memories: Dict[str, Memory] = {}
        self._load_all_memories()

    def _load_all_memories(self) -> None:
        """
        Load all memories from the memory directory into memory.
        
        The function only use in __init__ 
        """
        for memory_file in self.memory_dir.glob("*.md"):
            with open(memory_file, "r") as f:
                content: str = f.read()
                memory: Optional[Memory] = self._parse_formatter(content)
                if memory:
                    memory.file_name = memory_file.name
                    self.memories[memory.description] = memory
        
        if len(self.memories) > 0:
            logger.info(f"Loaded {len(self.memories)} memories from {self.memory_dir.relative_to(WORKDIR)}")

    def _parse_formatter(self, text: str) -> Optional[Memory]:
        """
        Parse a memory from the given text in the format of:
        ---
        name: {name}
        description: {description}
        type: {type}
        ---
        {content}

        file_name is not included in the text, it will be set in _load_all_memories function
        """
        # The regex pattern to match the memory format, with named groups for name, description and type
        pattern: str = r"^---\s*\n(.*?)\n---\s*\n(.*)"
        match: Optional[re.Match] = re.match(pattern, text.strip(), re.DOTALL)
        if not match:
            return None
        
        header: str = match.group(1)
        body: str = match.group(2)
        for line in header.splitlines():
            line = line.strip()
            if line.startswith("name:"):
                name: str = line[len("name:"):].strip()
            elif line.startswith("description:"):
                description: str = line[len("description:"):].strip()
            elif line.startswith("type:"):
                memory_type: MemoryType = line[len("type:"):].strip() # type: ignore
            
        return Memory(description=description, type=memory_type, content=body.strip(), file_name="")
    
    def build_memory_prompt(self) -> str:
        """
        Build a prompt string that summarizes all memories for the agent to use.
        """
        lines: List[str] = []
        for name, memory in self.memories.items():
            lines.append(f"### {name}: {memory.description}")
            if memory.content and len(memory.content) > 0:
                lines.append(memory.content)
                lines.append("")
            if len(lines) >= MAX_INDEX_LINES:
                break
        return "Memories (persistent across sessions)\n" + "\n".join(lines)

    async def save_memory(self, name: str, description: str, memory_type: MemoryType, content: str) -> str:
        """
        Save a memory to disk and memory
        
        return the operation result message
        """
        # safe means only contain letters, numbers, underscores and dashes
        safe_name: str = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
        if not safe_name or len(safe_name) == 0:
            return "Error: Invalid memory name. Memory name must contain at least one valid character (letters, numbers, underscores, or dashes)."
        
        file_name: str = f"{safe_name}.md"
        file_path: Path = self.memory_dir / file_name

        memory_frontmatter: str = "---\n" \
                                f"name: {name}\n" \
                                f"description: {description}\n" \
                                f"type: {memory_type}\n" \
                                "---\n" \
                                f"{content}\n" \
        

        print(memory_frontmatter)

        async with aiofiles.open(file_path, "w") as f:
            await f.write(memory_frontmatter.strip())

        # also keep it in memory for quick access, we can load it from disk when the agent start
        self.memories[name] = Memory(
            description=description, 
            type=memory_type, 
            content=content, 
            file_name=file_name
        )

        # rebuild the index with the new memory
        await self._rebuild_index()

        return f"Saved memory '{name}' [{memory_type}] to {file_path.relative_to(WORKDIR)}"
    
    async def delete_memory(self, name: str) -> str:
        """
        Delete a memory from disk and memory
        
        return the operation result message
        """
        memory: Optional[Memory] = self.memories.get(name)
        if not memory:
            return f"Error: Memory '{name}' not found."
        
        file_path: Path = self.memory_dir / memory.file_name
        if file_path.exists():
            await aiofiles.os.remove(file_path)
        
        self.memories.pop(name, None)

        # rebuild the index after deleting the memory
        await self._rebuild_index()

        return f"Deleted memory '{name}' and removed file {file_path.relative_to(WORKDIR)}"
    
    async def _rebuild_index(self) -> None:
        """
        Rebuild the index in MEMORY.md
        """
        lines: List[str] = []
        for name, memory in self.memories.items():
            lines.append(f"- {name}: {memory.description} [{memory.type}]")
            if len(lines) >= MAX_INDEX_LINES:
                break
        async with aiofiles.open(MEMORY_INDEX, "w") as f:
            await f.write("\n".join(lines) + "\n")

memory_manager: MemoryManager = MemoryManager()
