from pydantic import BaseModel
from typing import Dict, Any, Literal, Optional, List
from pathlib import Path
import json
import logging
import os
import asyncio.subprocess
from asyncio.subprocess import Process

from directory import WORKDIR

logger = logging.getLogger(__name__)

HOOK_TIMEOUT: int = 30 # seconds

class HookPayload(BaseModel):
    tool_name: str = ""
    tool_input: Dict[str, Any] = {}

class HookEvent(BaseModel):
    name: Literal["PreToolCall", "PostToolCall", "SessionStart"]
    payload: HookPayload

class HookDefinition(BaseModel):
    matcher: Optional[str] = None # the matched hooked tool name, also support * for all tools
    command: Optional[str] = None # the command to run when hooked

# a hook progarm returns
# - exit_code: Literal[0, 1, 2] # 0 for continue, 1 for reject, 2 for append message
# - stdout: str # if exit_code is 0, it can be a json string with optional fields: {
#       updateInput: for updating the tool input, 
#       additionalContext: for appending message to the conversation, 
#       permissionOverride: for overriding the permission of the tool call
#   }
# - stderr: str # If exit_code is 1, it should be the reason for blocking. 
#                 If exit_code is 2, it should be the message to append

class HookResponse(BaseModel):
    blocked: bool = False    # It's the hook blocked the execution ? only useful in PreToolCall event
    blocked_reason: Optional[str] = None # If blocked is True, it's the reason for blocking
    permission_override: Optional[str] = None # If the hook want to override the permission of the tool call, it can return the permission string here, now we don't use it
    messages: List[str] = [] # It's the messages that the hook want to append to the conversation, only work when blocked is False

class HookManager:
    def __init__(self, config_path: Optional[Path] = None):
        self.hooks: Dict[str, List[HookDefinition]] = {"PreToolCall": [], "PostToolCall": [], "SessionStart": []}
        config_path = config_path if config_path else WORKDIR / ".hooks.json"
        if config_path.exists():
            try:
                config: dict = json.loads(config_path.read_text())
                for event_name, hook_fns in config.items():
                    if event_name in self.hooks:
                        self.hooks[event_name] = [HookDefinition(**hook) for hook in config.get("hooks", {}).get(event_name, [])]
                logger.info(f"Loaded hooks configuration from {str(config_path)}")
            except json.JSONDecodeError:
                logger.error(f"Failed to parse hooks configuration from {str(config_path)}. Ensure it's a valid JSON file.")


    def register_hook(self, event_name: str, hook_fn):
        if event_name not in self.hooks:
            self.hooks[event_name] = []
        self.hooks[event_name].append(hook_fn)

    async def run_hooks(self, event: HookEvent) -> HookResponse:
        result: HookResponse = HookResponse()
        hooks: List[HookDefinition] = self.hooks.get(event.name, [])

        for hook in hooks:
            if hook.matcher and event.payload and hook.matcher != "*" and hook.matcher != event.payload.tool_name:
                continue  # Skip hooks that don't match the tool call
            
            if not hook.command:
                continue  # Skip hooks that don't have a command defined

            env: Dict[str, str] = os.environ.copy()
            if event.payload:
                env["HOOK_EVENT"] = event.name
                env["HOOK_TOOL_NAME"] = event.payload.tool_name
                env["HOOK_TOOL_INPUT"] = json.dumps(event.payload.tool_input, ensure_ascii=False)[:10000]

            try:
                r: Process = await asyncio.create_subprocess_shell(
                    hook.command,
                    shell=True,
                    cwd=WORKDIR,
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(r.communicate(), timeout=HOOK_TIMEOUT)
                if r.returncode == 0: # normal exit
                    stdout.strip()
                    try:
                        hook_output = json.loads(stdout) if stdout else {}
                        if "updateInput" in hook_output:
                            event.payload.tool_input = hook_output["updateInput"]
                        if "additionalContext" in hook_output:
                            result.messages.append(hook_output["additionalContext"])
                        if "permissionOverride" in hook_output:
                            result.permission_override = hook_output["permissionOverride"]
                    except json.JSONDecodeError:
                        logger.warning(f"Hook command did not return valid JSON: {hook.command}\nOutput: {stdout.decode().strip()}")
                elif r.returncode == 1: # blocked by hook
                    result.blocked = True # No other will change it, so if one hook blocked it, it's blocked forever
                    logger.info(f"Hook blocked the tool call: {hook.command}\nOutput: {stdout.decode().strip()}")
                    result.blocked_reason = stderr.decode().strip() if stderr else "Blocked by hook."
                elif r.returncode == 2: # append message
                    logger.info(f"Hook appended message to the conversation: {hook.command}\nMessage: {stderr.decode().strip()}")
                    result.messages.append(stderr.decode().strip())
                logger.error(f"Hook command timed out: {hook.command}")
            except Exception as e:
                logger.error(f"Error running hook command '{hook.command}': {e}")

        return result
    
hook_manager: HookManager = HookManager()