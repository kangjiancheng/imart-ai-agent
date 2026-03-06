# Stub — read-only SQL query tool
from langchain_core.tools import tool


@tool
def database_query(sql: str) -> str:
    """Run a READ-ONLY SQL query against the internal database.
    Never accepts INSERT, UPDATE, DELETE, or DROP statements."""
    return "Database query tool is not configured in this environment."
