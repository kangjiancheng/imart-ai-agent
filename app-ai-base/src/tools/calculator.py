import ast
import operator

from langchain_core.tools import tool

# Whitelist of safe AST node types → Python operator functions.
# Any expression node not in this dict is rejected, preventing code injection.
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(node):
    """
    Recursively evaluate an AST node using the _SAFE_OPS whitelist.
    Raises ValueError for any node type not in the whitelist.
    """
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        op = _SAFE_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operation: {type(node.op).__name__}")
        return op(_safe_eval(node.left), _safe_eval(node.right))
    elif isinstance(node, ast.UnaryOp):
        op = _SAFE_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operation: {type(node.op).__name__}")
        return op(_safe_eval(node.operand))
    else:
        raise ValueError(f"Unsupported expression type: {type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Use for arithmetic, percentages,
    and unit conversions. Example inputs: '15 * 1.08', '(100 - 20) / 4'."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return str(result)
    except Exception as e:
        return f"Calculation error: {str(e)}"
