from enum import Enum


class FinishReason(str, Enum):
    STOP = "stop"
    FUNCTION_CALL = "function_call"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
