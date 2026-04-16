from typing import List, Literal
from pydantic import BaseModel

PLAN_REMINDER_THRESHOLD: int = 3  # Number of rounds without updates before sending a reminder

class TodoItem(BaseModel):
    content: str
    status: Literal["pending", "in_progress", "completed"] = "pending"
    activeForm: str = ""  # Optional field to indicate if there's an active form associated with this todo item

class TodoManager:
    def __init__(self):
        self.todos: List[TodoItem] = []
        self.rounds_since_update: int = 0

    def update(self, todos: List[TodoItem]) -> str:
        """Update the todo list with new items and return a rendered string representation."""
        validated_todos: List[TodoItem] = []
        in_progress_count: int = 0
        for todo in todos:
            status = todo.status
            if status in ["pending", "in_progress", "completed"]:
                if status == "in_progress":
                    in_progress_count += 1
                validated_todos.append(TodoItem(
                    content=todo.content,
                    status=status,
                    activeForm=todo.activeForm,
                ))
            else:
                raise ValueError(f"Invalid status '{status}' for todo item: {todo.content}")
        if in_progress_count > 1:
            raise ValueError("Only one todo item can be in progress at a time.")
        self.todos = validated_todos
        self.rounds_since_update = 0
        return self.render()
    
    def render(self) -> str:
        """Render the todo list as a string."""
        if not self.todos:
            return ""
        rendered_items = []
        for todo in self.todos:
            status_symbol = {
                "pending": "[ ]",
                "in_progress": "[~]",
                "completed": "[o]",
            }.get(todo.status)
            rendered_items.append(f"{status_symbol} {todo.content}")

        completed: int = sum(1 for todo in self.todos if todo.status == "completed")
        rendered_items.append(f"\n{completed}/{len(self.todos)} completed")
        return "\n".join(rendered_items)
    
    def reminder(self) -> str | None:
        """Return a reminder string if there are pending or in-progress items that haven't been updated for 3 rounds."""
        if self.rounds_since_update >= PLAN_REMINDER_THRESHOLD:
            pending_or_in_progress = [todo for todo in self.todos if todo.status in ["pending", "in_progress"]]
            if pending_or_in_progress:
                return "<reminder>Refresh your current plan before continuing.</reminder>"
        return None
    
    def note_round_without_update(self) -> None:
        self.rounds_since_update += 1

todo_manager = TodoManager()