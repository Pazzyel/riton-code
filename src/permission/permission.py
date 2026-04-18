from typing import List, TypeAlias, Optional, Tuple
from typing_extensions import Literal
from pydantic import BaseModel
from fnmatch import fnmatch
import json

from permission.bash_security import bash_security_validator

PermissionMode: TypeAlias = Literal["default", "plan", "auto"]
PermissionBehavior: TypeAlias = Literal["allow", "deny", "ask"]

MAX_CONSTITUTIVE_DENIAL: int = 3

READ_ONLY_TOOLS: List[str] = ["read_file"]
WRITE_TOOLS: List[str] = ["write_file", "edit_file"]

class PermissionRule(BaseModel):
    """
    Represents a permission rule for a tool.
    """
    tool: str
    behavior: PermissionBehavior
    path: Optional[str] = None
    content: Optional[str] = None

DEFAULT_RULES: List[PermissionRule] = [
    PermissionRule(tool="bash", content="rm -rf /", behavior="deny"),
    PermissionRule(tool="bash", content="sudo *", behavior="deny"),
    PermissionRule(tool="read_file", path="*", behavior="allow"),
]

class PermissionResult(BaseModel):
    """
    Represents the result of evaluating permission rules for a tool call.
    """
    behavior: PermissionBehavior
    reason: Optional[str] = None

class PermissionManager:
    """
    Manages permission rules and evaluates them for tool calls.
    """
    def __init__(self, mode: PermissionMode = "default", rules: Optional[List[PermissionRule]] = None):
        self.mode = mode
        self.rules: List[PermissionRule] = rules if rules is not None else DEFAULT_RULES
        self.constitutive_denials: int = 0
        self.max_constitutive_denials: int = MAX_CONSTITUTIVE_DENIAL

    def add_rule(self, rule: PermissionRule):
        self.rules.append(rule)

    def check(self, tool_name: str, tool_args: dict) -> PermissionResult:
        """Check the permission for a given tool call based on the defined rules and mode."""
        # 1. Special handling for bash commands with security validation
        if tool_name == "bash":
            bash_command: str = tool_args.get("command", "")
            failures: List[Tuple[str, str]] = bash_security_validator.validate(bash_command)
            if failures:
                server_failure = [f for f in failures if f[0] in bash_security_validator.SERVER_COMMANDS]
                if server_failure:
                    return PermissionResult(behavior="deny", reason=bash_security_validator.describe_failures(bash_command))
                else:
                    return PermissionResult(behavior="ask", reason=bash_security_validator.describe_failures(bash_command))

        # 2. Handle deny rules first
        for rule in self.rules:
            if rule.behavior != "deny":
                continue
            if self._matches(rule, tool_name, tool_args):
                self.constitutive_denials += 1
                if self.constitutive_denials >= self.max_constitutive_denials:
                    return PermissionResult(behavior="deny", reason=f"Constitutive denial threshold reached for rule: {rule}, recommended to switch to 'plan' mode for better performance, or reclarify the target.")
                else:
                    return PermissionResult(behavior="deny", reason=f"Deny by the rule: {rule}")

        # 3. Handle by mode
        if self.mode == "plan":
            if tool_name in WRITE_TOOLS:
                return PermissionResult(behavior="deny", reason=f"All write tool included '{tool_name}' is not allowed in 'plan' mode.")
            else:
                return PermissionResult(behavior="allow", reason=f"All read-only tools included '{tool_name}' are allowed in 'plan' mode.")
        if self.mode == "auto":
            if tool_name in READ_ONLY_TOOLS:
                return PermissionResult(behavior="allow", reason=f"All read-only tools included '{tool_name}' are auto approved in 'auto' mode.")
            # else for ask

        # 4. Check allow rules
        for rule in self.rules:
            if rule.behavior != "allow":
                continue
            if self._matches(rule, tool_name, tool_args):
                self.constitutive_denials = 0  # reset constitutive denial count on allow
                return PermissionResult(behavior="allow", reason=f"Allow by the rule: {rule}") 

        # If no rules matched, apply ask
        return PermissionResult(behavior="ask", reason=f"No matching rule for {tool_name} found, so asking for confirmation."  )
    
    def _matches(self, rule: PermissionRule, tool_name: str, tool_args: dict) -> bool:
        """
        Match tool_name, tool_path, and tool_content against the rule's specifications.
        """
        if rule.tool != "*" and rule.tool != tool_name:
            return False
        if rule.path is not None and rule.path != "*":
            tool_path = tool_args.get("path", "")
            if not fnmatch(tool_path, rule.path):
                return False
        if rule.content is not None:
            tool_content = tool_args.get("content", "")
            if not fnmatch(tool_content, rule.content):
                return False
        return True
    
    def ask_user(self, tool_name: str, tool_args: dict) -> bool:
        """Ask the user, y/n/always, for permission to execute a tool call."""
        tool_args_preview: str = json.dumps(tool_args)[:200]  # limit preview length
        print(f"\n  [Permission] {tool_name} with args {tool_args_preview}  ")
        try:
            user_input: str = input("Allow? (y/n/always): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            return False
        
        if user_input == "always":
            # Add an all allow rule for this tool
            allowed_rule = PermissionRule(
                tool=tool_name,
                behavior="allow",
                path="*",
            )
            self.add_rule(allowed_rule)
            self.constitutive_denials = 0  # reset constitutive denial count on new allow rule
            return True
        if user_input == "y":
            self.constitutive_denials = 0  # reset constitutive denial count on allow
            return True
        self.constitutive_denials += 1
        if self.constitutive_denials >= self.max_constitutive_denials:
            print(f"Constitutive denial threshold reached, recommended to switch to 'plan' mode for better performance, or reclarify the target.")
        return False
    
permission_manager = PermissionManager()