from enum import Enum


class IntentType(str, Enum):
    ACTIONABLE = "ACTIONABLE"
    GREETING = "GREETING"
    UNKNOWN = "UNKNOWN"
