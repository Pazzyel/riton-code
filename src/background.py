from threading import Lock, Thread
import time

from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from typing_extensions import Literal
from pydantic import BaseModel

MAX_OUTPUT_LENGTH = 50000
MAX_PREVIEW_LENGTH = 50000
_background_counter: int = 0  # Global counter for background tasks

class RuntimeTaskRecord(BaseModel):
    id: str
    command: str
    status: Literal["running", "completed", "timeout", "error"] = "running"
    result: str = ""
    start_at: Optional[float] = None
    finish_at: Optional[float] = None
    result_preview: str = ""
    output_file: str = "" # No use

class Notification(BaseModel):
    type: str
    task_id: str
    status: str
    command: str
    preview: str
    output_file: str # No use

class BackgroundManager:
    def __init__(self):
        
        # self.dir: Path = dir
        self.runtime_tasks: Dict[str, RuntimeTaskRecord] = {}
        # self.notification_queue: List[Notification] = []
        self._lock: Dict[str, Lock] = {}

    def is_slow_operation(self, tool_name: str, tool_input: Dict[str, Any]) -> bool:
            """Determine if the command is likely to be a slow operation."""
            if tool_name != "bash" and tool_name != "sh":
                return False
            command: str = tool_input.get("command", "").lower()
            slow_keywords: List[str] = ["install", "build", "test", "deploy", "compile",
                         "docker build", "pip install", "npm install",
                         "cargo build", "pytest", "make"]
            return any(keyword in command for keyword in slow_keywords)
    
    def should_run_in_background(self, tool_name: str, tool_input: Dict[str, Any]) -> bool:
        """Determine if the command should be run in the background."""
        # if the tool_input explicitly requests background execution, honor that
        # otherwise check if it's a slow operation
        if tool_input.get("run_in_background", False) is True:
            return True
        return self.is_slow_operation(tool_name, tool_input)

    def start_background_task(self, handler: Callable, tool_input: Dict[str, Any]) -> str:
        """Start a background task for the given tool and input. Return the task ID."""
        global _background_counter
        _background_counter += 1
        background_task_id: str = f"bg_task_{_background_counter}"

        def worker():
            try:
                result: Any = handler(**tool_input)
            except Exception as e:
                result = f"Error: {e}"
            
            with self._lock.get(background_task_id, Lock()):
                self.runtime_tasks[background_task_id].result = str(result)
                if str(result).startswith("Error: Command timed out"):
                    self.runtime_tasks[background_task_id].status = "timeout"
                elif str(result).startswith("Error:"):
                    self.runtime_tasks[background_task_id].status = "error"
                else:
                    self.runtime_tasks[background_task_id].status = "completed"
                self.runtime_tasks[background_task_id].finish_at = time.time()
                self.runtime_tasks[background_task_id].result_preview = self._preview(str(result))

        with self._lock.setdefault(background_task_id, Lock()):
            self.runtime_tasks[background_task_id] = RuntimeTaskRecord(
                id=background_task_id,
                command=tool_input.get("command", ""),
                status="running",
                start_at=time.time(),
            )

        thread: Thread = Thread(target=worker, daemon=True)
        thread.start()
        return background_task_id

    def _collect_background_task_result(self) -> List[RuntimeTaskRecord]:
        """Collect the result of a background task if it's completed."""
        completed_tasks: List[RuntimeTaskRecord] = []
        for background_task_id, task_record in self.runtime_tasks.items():
            with self._lock.get(background_task_id, Lock()):
                if task_record.status in ["completed", "timeout", "error"]:
                    completed_tasks.append(task_record)
        # When finding completed tasks, remove them from the runtime_tasks dictionary
        # If a task is completed, no other thread should be able to access it, so it's safe to remove it from the dictionary
        for task in completed_tasks:
            del self.runtime_tasks[task.id]
            del self._lock[task.id]  # Remove the lock for the completed task
                
        return completed_tasks

    def get_background_task_notification(self) -> str:
        """Get a notification string for any completed background tasks."""
        completed_tasks: List[RuntimeTaskRecord] = self._collect_background_task_result()
        if not completed_tasks:
            return ""
        notifications: str = "<task_notifications>\n"
        for task in completed_tasks:
            notifications += (
                f"  <task_notification>\n"
                f"      <task_id>{task.id}</task_id>\n"
                f"      <status>completed</status>\n"
                f"      <command>{task.command}</command>\n"
                f"      <summary>{task.result_preview}</summary>\n"
                f"  </task_notification>\n"
            )
        notifications += "</task_notifications>\n"
        return notifications


    # def _record_path(self, task_id: str) -> Path:
    #     return self.dir / f"{task_id}.json"

    # def _output_path(self, task_id: str) -> Path:
    #     return self.dir / f"{task_id}.log"
    
    # async def _save_task(self, task: RuntimeTaskRecord):
    #     """Save the runtime task record to a JSON file."""
    #     record_path: Path = self._record_path(task.id)
    #     async with aiofiles.open(record_path, "w") as f:
    #         await f.write(task.model_dump_json(indent=4))

    def _preview(self, output: str, max_length: int = MAX_PREVIEW_LENGTH) -> str:
        """Generate a preview of the output."""
        compact: str = " ".join((output or "(no output)").split())
        return compact[:max_length] + ("..." if len(compact) > max_length else "")
    
    # async def __execute_task(self, task_id: str, command: str):
    #     """Execute the task command and update the record."""
    #     try:
    #         result: CompletedProcess = subprocess.run(
    #             command, shell=True, cwd=WORKDIR,
    #             capture_output=True, text=True, timeout=MAX_SUBPROCESS_TIME)
    #         output: str = (result.stdout + result.stderr).strip()[:MAX_OUTPUT_LENGTH]
    #         status: str = "completed"
    #     except subprocess.TimeoutExpired as e:
    #         output = f"Error: timeout after {MAX_SUBPROCESS_TIME} seconds."
    #         status = "timeout"
    #     except Exception as e:
    #         output = f"Error: {str(e)}"
    #         status = "error"
    #     final_output: str = output or "(no output)"
    #     preview: str = self.__preview(final_output)
    #     output_path: Path = self.__output_path(task_id)
    #     async with aiofiles.open(output_path, "w") as f:
    #         await f.write(final_output)
    #     task: RuntimeTaskRecord = self.runtime_tasks[task_id]
    #     task.status = status
    #     task.output_file = str(output_path.relative_to(self.dir))
    #     task.result = final_output
    #     task.result_preview = preview
    #     task.finish_at = time.time()
    #     with self._lock:
    #         self.notification_queue.append(Notification(
    #             type="task_update",
    #             task_id=task_id,
    #             status=status,
    #             command=command,
    #             preview=preview,
    #             output_file=str(output_path.relative_to(self.dir)),
    #         ))

    # async def __run(self, command: str):
    #     """Create a new task record and start execution in a background thread."""
    #     task_id: str = str(uuid.uuid4())[:8]
    #     output_path: Path = self.__output_path(task_id)
    #     task: RuntimeTaskRecord = RuntimeTaskRecord(
    #         id=task_id,
    #         status="running",
    #         command=command,
    #         start_at=time.time(),
    #         output_file=str(output_path.relative_to(self.dir)),
    #     )
    #     self.runtime_tasks[task_id] = task
    #     await self.__save_task(task)
    #     thread: Thread = threading.Thread(
    #         target=self.__execute_task, args=(task_id, command), daemon=True
    #     )
    #     thread.start()
    #     return (
    #         f"Background task {task_id} started: {command[:80]} "
    #         f"(output_file={output_path.relative_to(self.dir)})"
    #     )

BACKGROUND_MANAGER: BackgroundManager = BackgroundManager()