# Stub — sandboxed subprocess code execution
# Enable with care: requires proper subprocess isolation (see architecture doc §13)
from langchain_core.tools import tool


@tool
def code_exec(code: str) -> str:
    """Execute Python code in a sandboxed subprocess and return stdout.
    Use for data analysis, formatting, or computation that needs real code."""
    return "Code execution is not enabled in this environment."
