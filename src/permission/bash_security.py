from typing import List, Tuple
import re

class BashSecurityValidator:
    VALIDATORS: List[Tuple[str,str]] = [
        ("shell_metchar", r"[;&|`$]"), # Prevent ; & | ` $ which can be used for command chaining
        ("rm_rf", r"\brm\s+(-[a-zA-Z]*)?r"), # Prevent rm -rf and variants
        ("command_substitution", r"\$\("), # Prevent command substitution which can execute arbitrary commands
        ("ifs_injection", r"\bIFS\s*="), # Prevent IFS injection which can change how commands are parsed
        ("sudo", r"\bsudo\b"), # Prevent use of sudo which can escalate privileges
    ]

    SERVER_COMMANDS: List[str] = ["sudo", "rm_rf"]

    def validate(self, command: str) -> List[Tuple[str, str]]:
        """Validate a bash command against known unsafe patterns."""
        failures: List[Tuple[str, str]] = []
        for name, pattern in self.VALIDATORS:
            if re.search(pattern, command):
                failures.append((name, pattern))
        return failures
    
    def is_safe(self, command: str) -> bool:
        """Check if a bash command is safe to execute."""
        return len(self.validate(command)) == 0
    
    def describe_failures(self, command: str) -> str:
        """Describe the reasons why a bash command is considered unsafe."""
        failures = self.validate(command)
        if not failures:
            return "Command is safe."
        else:
            parts: List[str] = [f"{name} (pattern: {pattern})" for name, pattern in failures]
            return f"Command is potentially unsafe due to: {', '.join(parts)}."
        
bash_security_validator = BashSecurityValidator()