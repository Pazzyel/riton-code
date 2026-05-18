from typing import List, Optional
from typing_extensions import Literal

from pydantic import BaseModel
import aiofiles
import aiopath
from pathlib import Path

from directory import TASKS_DIR

class Task(BaseModel):
    id: int
    subject: str
    description: str = ""
    status: Literal["pending", "in_progress", "completed", "deleted"] = "pending"
    blocked_by: List[int] = []
    blocks: List[int] = []
    owner: str = ""

def is_ready(task: Task) -> bool:
    """Check if a task is ready to be worked on (i.e., all its dependencies are completed)."""
    return task.status == "pending" and len(task.blocked_by) == 0

class TaskManager:
    def __init__(self, task_dir: Path):
        self.task_dir = task_dir
        self.task_dir.mkdir(parents=True, exist_ok=True)
        self._next_id = self._get_max_id() + 1

    def _get_max_id(self) -> int:
        """Get the maximum task ID currently in the task directory."""
        ids: List[int] = [int(f.stem.split("_")[1]) for f in self.task_dir.glob("task_*.json")]
        return max(ids) if ids else 0
    
    def _get_next_id(self) -> int:
        """Get the next available task ID."""
        current_id = self._next_id
        self._next_id += 1
        return current_id
    
    async def _save(self, task: Task) -> None:
        """Save a task to the task directory."""
        task_path: Path = self.task_dir / f"task_{task.id}.json"
        async with aiofiles.open(task_path, "w") as f:
            await f.write(task.model_dump_json())

    async def create(self, subject: str, description: str = "", owner: str = "") -> str:
        """Create and save a new task."""
        task: Task = Task(
            id=self._get_next_id(),
            subject=subject,
            description=description,
            owner=owner
        )
        await self._save(task)
        return task.model_dump_json()
    
    async def _load(self, task_id: int) -> Task:
        """Load a task from the task directory."""
        task_path: Path = self.task_dir / f"task_{task_id}.json"
        async with aiofiles.open(task_path, "r") as f:
            data = await f.read()
            return Task.model_validate_json(data)
        
    async def get(self, task_id: int) -> str:
        """Get a task by its ID."""
        return (await self._load(task_id)).model_dump_json()
    
    async def clear_dependencies(self, task_id: int) -> None:
        """
        Clear all dependencies of a task.
        
        This will remove the task from the blocked_by list of any tasks that depend on it
        """
        async for f in aiopath.AsyncPath(self.task_dir).glob("task_*.json"):
            async with aiofiles.open(f, "r") as file:
                data = await file.read()
                task = Task.model_validate_json(data)
                if task_id in task.blocked_by:
                    task.blocked_by.remove(task_id)
                    await self._save(task)
    
    async def update_dependencies(self, 
                                  task_id: int, 
                                  status: Optional[Literal["pending", "in_progress", "completed", "deleted"]] = None,
                                  owner: Optional[str] = None,
                                  add_blocked_by: Optional[List[int]] = None,
                                  add_blocks: Optional[List[int]] = None
                                ) -> str:
        """Update the dependencies of a task."""
        task = await self._load(task_id)
        if status is not None:
            task.status = status
            if status == "completed":
                await self.clear_dependencies(task_id)
        if owner is not None:
            task.owner = owner
        if add_blocked_by is not None:
            task.blocked_by = list(set(task.blocked_by).union(set(add_blocked_by)))
        if add_blocks is not None:
            task.blocks = list(set(task.blocks).union(set(add_blocks)))
        await self._save(task)
        return task.model_dump_json()
    
    async def list_all(self) -> str:
        """List all tasks in the task directory."""
        tasks: List[Task] = []
        async for f in aiopath.AsyncPath(self.task_dir).glob("task_*.json"):
            async with aiofiles.open(f, "r") as file:
                data = await file.read()
                tasks.append(Task.model_validate_json(data))
        if not tasks:
            return "No tasks found."
        lines: List[str] = []
        for task in tasks:
            marker: str = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]", "deleted": "[-]"}.get(task.status, "[?]")
            blocked: str = f" (blocked by: {task.blocked_by})" if task.blocked_by else ""
            owner: str = f" owner={task.owner}" if task.owner else ""
            lines.append(f"{marker} #{task.id}: {task.subject}{owner}{blocked}")
        return "\n".join(lines)
    
TASKS_MANAGER: TaskManager = TaskManager(task_dir=TASKS_DIR)