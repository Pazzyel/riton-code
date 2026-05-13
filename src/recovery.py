from typing import Optional, Literal
from pydantic import BaseModel

CONTINUE_MESSAGE: str = (
    "Output limit hit. Continue directly from where you stopped. "
    "Do not restart or repeat."
)

BACKOFF_BASE_DELAY = 1.0  # seconds
BACKOFF_MAX_DELAY = 32.0  # seconds

class RecoveryType(BaseModel):
    type: Literal["success", "continue", "compact", "backoff", "fail"]
    message: str

def choose_recovery(stop_reason: Optional[str], error_text: Optional[str]) -> RecoveryType:
    if stop_reason == "length":
        return RecoveryType(type="continue", message="output truncated")
    if not error_text:
        return RecoveryType(type="success", message="No errors, normal completion")
    if error_text and "long" in error_text:
        return RecoveryType(type="compact", message="context too long")
    if error_text and any(keyword in error_text for keyword in ["timeout", "rate", "unavailable", "connection"]):
        return RecoveryType(type="backoff", message="transient transport failure")
    return RecoveryType(type="fail", message="unrecoverable error")

def backoff_delay(retry_count: int) -> int:
    delay = min(BACKOFF_BASE_DELAY * (2 ** retry_count), BACKOFF_MAX_DELAY)
    return delay