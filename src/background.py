from threading import Lock, RLock, Thread
import asyncio
import inspect
import logging
import time
from typing import Dict, List, Optional, Any, Callable, Awaitable
from typing_extensions import Literal
from pydantic import BaseModel

logger = logging.getLogger(__name__)

MAX_OUTPUT_LENGTH = 50000
MAX_PREVIEW_LENGTH = 50000
_background_counter: int = 0  # Global counter for background tasks

class RuntimeTaskRecord(BaseModel):
    id: str
    agent_id: str
    tool_name: str
    tool_input: Dict[str, Any]
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
        self._manager_lock: RLock = RLock()
        self._change_callback: Optional[Callable[[], None]] = None
        self._completion_callback: Optional[Callable[[RuntimeTaskRecord], None]] = None

    def set_callbacks(
        self,
        change_callback: Optional[Callable[[], None]],
        completion_callback: Optional[Callable[[RuntimeTaskRecord], None]] = None,
    ) -> None:
        self._change_callback = change_callback
        self._completion_callback = completion_callback

    def snapshot(self) -> List[RuntimeTaskRecord]:
        with self._manager_lock:
            tasks: List[RuntimeTaskRecord] = [
                task.model_copy(deep=True) for task in self.runtime_tasks.values()
            ]
        return tasks

    def restore(self, tasks: List[RuntimeTaskRecord]) -> None:
        global _background_counter
        with self._manager_lock:
            self.runtime_tasks = {
                task.id: task.model_copy(deep=True) for task in tasks
            }
            self._lock = {task.id: Lock() for task in tasks}
            numeric_ids: List[int] = []
            for task in tasks:
                if task.id.startswith("bg_task_") and task.id[8:].isdigit():
                    numeric_ids.append(int(task.id[8:]))
            if numeric_ids:
                _background_counter = max(_background_counter, max(numeric_ids))

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
        # return self.is_slow_operation(tool_name, tool_input)
        return False

    def start_background_task(
        self,
        handler: Callable,
        tool_name: str,
        tool_input: Dict[str, Any],
        agent_id: str,
    ) -> str:
        """Start a background task for the given tool and input. Return the task ID."""
        global _background_counter
        _background_counter += 1
        background_task_id: str = f"bg_task_{_background_counter}"

        task_record: RuntimeTaskRecord = RuntimeTaskRecord(
            id=background_task_id,
            agent_id=agent_id,
            tool_name=tool_name,
            tool_input=dict(tool_input),
            command=str(tool_input.get("command", "")),
            status="running",
            start_at=time.time(),
        )
        with self._manager_lock:
            self._lock[background_task_id] = Lock()
            self.runtime_tasks[background_task_id] = task_record
        self._notify_change()
        self._launch_task(background_task_id, handler)
        return background_task_id

    def _launch_task(self, background_task_id: str, handler: Callable) -> None:
        """启动后台任务，正常启动或恢复启动都是这条路径"""
        def worker() -> None:
            # A isolation thread not belong to the event loop
            with self._manager_lock:
                task_record: RuntimeTaskRecord = self.runtime_tasks[background_task_id]
                tool_input: Dict[str, Any] = dict(task_record.tool_input)
            try:
                logger.debug(
                    f"Background task {background_task_id} started for agent {task_record.agent_id}."
                )
                result: Any = handler(**tool_input)
                if inspect.isawaitable(result):
                    # If the handler is a coroutine, run it in the event loop
                    awaitable_result: Awaitable = result
                    async def run_async():
                        return await awaitable_result
                    result = asyncio.run(run_async())
                    
            except Exception as e:
                result = f"Error: {e}"
            
            with self._manager_lock, self._lock.get(background_task_id, Lock()):
                current_task: RuntimeTaskRecord = self.runtime_tasks[background_task_id]
                current_task.result = str(result)
                if str(result).startswith("Error: Command timed out"):
                    current_task.status = "timeout"
                elif str(result).startswith("Error:"):
                    current_task.status = "error"
                else:
                    current_task.status = "completed"
                current_task.finish_at = time.time()
                current_task.result_preview = self._preview(str(result))
                completed_task: RuntimeTaskRecord = current_task.model_copy(deep=True)
                logger.debug(f"Background task {background_task_id} completed.")
            self._notify_change()
            if self._completion_callback is not None:
                self._completion_callback(completed_task)

        thread: Thread = Thread(target=worker, daemon=True)
        thread.start()

    def resume_running_tasks(
        self,
        handler_resolver: Callable[[str, str], Optional[Callable]],
    ) -> None:
        """恢复正在执行的后台任务，也就是找到running的任务再launch一遍"""
        tasks: List[RuntimeTaskRecord] = self.snapshot()
        for task in tasks:
            if task.status != "running":
                continue
            handler: Optional[Callable] = handler_resolver(task.tool_name, task.agent_id)
            if handler is None:
                with self._manager_lock, self._lock[task.id]:
                    current_task: RuntimeTaskRecord = self.runtime_tasks[task.id]
                    current_task.status = "error"
                    current_task.result = f"Error: Unknown tool call: {task.tool_name}"
                    current_task.result_preview = self._preview(current_task.result)
                    current_task.finish_at = time.time()
                self._notify_change()
                continue
            self._launch_task(task.id, handler)

    def _notify_change(self) -> None:
        if self._change_callback is not None:
            self._change_callback()

    def _collect_background_task_result(self, agent_id: str) -> List[RuntimeTaskRecord]:
        """Collect the result of a background task if it's completed."""
        completed_tasks: List[RuntimeTaskRecord] = []
        with self._manager_lock:
            for background_task_id, task_record in self.runtime_tasks.items():
                with self._lock.get(background_task_id, Lock()):
                    if task_record.status in ["completed", "timeout", "error"] and task_record.agent_id == agent_id:
                        completed_tasks.append(task_record.model_copy(deep=True))
        # When finding completed tasks, remove them from the runtime_tasks dictionary
        # If a task is completed, no other thread should be able to access it, so it's safe to remove it from the dictionary
            for task in completed_tasks:
                del self.runtime_tasks[task.id]
                del self._lock[task.id]  # Remove the lock for the completed task
                
        return completed_tasks

    def get_background_task_notification(self, agent_id: str) -> str:
        """Get a notification string for any completed background tasks."""
        completed_tasks: List[RuntimeTaskRecord] = self._collect_background_task_result(agent_id)
        logger.info(f"Collected {len(completed_tasks)} completed background tasks for agent {agent_id}.")
        if not completed_tasks:
            return ""
        notifications: str = "<task_notifications>\n"
        for task in completed_tasks:
            notifications += (
                f"  <task_notification>\n"
                f"      <task_id>{task.id}</task_id>\n"
                f"      <status>{task.status}</status>\n"
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
