# src/tools/calculator.py
#
# ─────────────────────────────────────────────────────────────────────────────
# WHAT IS THIS TOOL?
#
# This file defines the `calculator` tool — Claude calls it to evaluate math.
# Example: user asks "What is 15% tip on a $47.80 meal?"
#   Claude calls calculator(expression="47.80 * 0.15")
#   This returns "7.17" and Claude uses that in its answer.
#
# WHY NOT JUST USE Python's eval()?
#   eval("47.80 * 0.15") would work for math, BUT eval() can execute ANY Python:
#     eval("__import__('os').system('rm -rf /')")  ← catastrophic!
#   Since Claude provides the expression string based on user input,
#   a malicious user could craft a message that tricks Claude into calling
#   calculator(expression="__import__('os').system('rm -rf /')")
#
# THE SAFE SOLUTION — AST-based evaluation:
#   1. Parse the expression string into an Abstract Syntax Tree (AST)
#   2. Walk the tree manually, only allowing specific node types
#   3. Reject ANYTHING that isn't a number or a safe math operation
#   This completely eliminates the code injection risk.
# ─────────────────────────────────────────────────────────────────────────────

import ast
# ast = Python's built-in Abstract Syntax Tree module.
# ast.parse("2 + 3 * 4") → a tree structure representing the expression:
#   BinOp(
#     left=Constant(2),
#     op=Add(),
#     right=BinOp(left=Constant(3), op=Mult(), right=Constant(4))
#   )
# Instead of running the code, we walk this tree structure ourselves.

import operator
# operator = Python's built-in module with functions for math operations.
# operator.add(2, 3)  = 5     (same as 2 + 3)
# operator.mul(4, 5)  = 20    (same as 4 * 5)
# operator.neg(-7)    = 7     (negation, same as -(-7))
# Using these functions instead of inline math makes the code data-driven.

from langchain_core.tools import tool
# @tool decorator — see web_search.py for full explanation.


# ── Safe operations whitelist ─────────────────────────────────────────────────
# Only allow safe math operations — no eval() on user input.
#
# PYTHON CONCEPT — dict (dictionary):
#   A dict maps keys to values: { key: value, key: value, ... }
#   In TypeScript: { [key: Type]: ValueType } or Map<Key, Value>
#
# _SAFE_OPS maps AST NODE TYPES → Python operator functions:
#   ast.Add  → operator.add   (handles "+" in expressions)
#   ast.Sub  → operator.sub   (handles "-" in expressions)
#   ast.Mult → operator.mul   (handles "*" in expressions)
#   etc.
#
# The leading underscore `_SAFE_OPS` signals "private to this module".
# Other files that import calculator.py cannot accidentally use _SAFE_OPS.
_SAFE_OPS = {
    ast.Add: operator.add,      # binary +   e.g. "3 + 4"
    ast.Sub: operator.sub,      # binary -   e.g. "10 - 3"
    ast.Mult: operator.mul,     # *          e.g. "5 * 6"
    ast.Div: operator.truediv,  # /          e.g. "10 / 3" → 3.333...
    ast.Pow: operator.pow,      # **         e.g. "2 ** 8" → 256
    ast.USub: operator.neg,     # unary -    e.g. "-5" (negative number)
}
# Everything NOT in this dict is blocked.
# Attempting "import os" or "open('file')" produces AST nodes not in this dict,
# so _safe_eval raises ValueError before executing anything.


# ── _safe_eval: recursive AST tree walker ─────────────────────────────────────
# PYTHON CONCEPT — recursion:
#   A function that calls ITSELF on smaller sub-problems.
#   "2 + 3 * 4" is a tree:
#     BinOp(left=2, op=+, right=BinOp(left=3, op=*, right=4))
#   _safe_eval(outer BinOp)
#     calls _safe_eval(2) → returns 2
#     calls _safe_eval(inner BinOp)
#       calls _safe_eval(3) → returns 3
#       calls _safe_eval(4) → returns 4
#       returns 3 * 4 = 12
#     returns 2 + 12 = 14
#
# PYTHON CONCEPT — isinstance():
#   isinstance(node, ast.Constant) = checks if `node` is an instance of the
#   `ast.Constant` class (i.e., is a number/string literal in the AST).
#   In TypeScript: node instanceof AstConstant

def _safe_eval(node):
    # CASE 1: Constant (a literal number like 42 or 3.14)
    if isinstance(node, ast.Constant):
        # node.value is the actual Python number value.
        # Return it directly — no operation needed.
        return node.value

    # CASE 2: Binary operation (two operands + one operator)
    # Examples: "2 + 3", "10 / 4", "2 ** 8"
    elif isinstance(node, ast.BinOp):
        # PYTHON CONCEPT — type() vs isinstance():
        #   type(node.op) returns the EXACT class of the operator node.
        #   e.g., type(node.op) is ast.Add for "+"
        #   We use type() here (not isinstance) because we need to look up
        #   the exact class in our _SAFE_OPS dictionary.
        op = _SAFE_OPS.get(type(node.op))
        # .get(key) returns None if the key is not in the dict.
        # If the operator is not in _SAFE_OPS (e.g., ast.BitOr for "|"),
        # op will be None and we raise ValueError below.

        if op is None:
            # type(node.op).__name__ gets a human-readable name for the node type.
            # e.g., "BitOr" for bitwise OR — which is not a safe math operation.
            raise ValueError(f"Unsupported operation: {type(node.op).__name__}")

        # Recursively evaluate left and right sub-expressions, then apply the operator.
        # op(_safe_eval(node.left), _safe_eval(node.right))
        #   = call the operator function with two evaluated sub-results
        #   Example: operator.add(_safe_eval(left=2), _safe_eval(right=3)) = 5
        return op(_safe_eval(node.left), _safe_eval(node.right))

    # CASE 3: Unary operation (one operand)
    # Example: "-5" (negative sign before a number)
    elif isinstance(node, ast.UnaryOp):
        op = _SAFE_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operation: {type(node.op).__name__}")
        # node.operand = the single sub-expression being operated on
        return op(_safe_eval(node.operand))

    # CASE 4: Anything else is blocked — variable names, function calls, etc.
    else:
        # If the expression contains a variable name like `x`, a function call
        # like `open(...)`, or any other non-math construct, we reject it here.
        raise ValueError(f"Unsupported expression type: {type(node).__name__}")


# ── calculator tool ─────────────────────────────────────────────────────────
# @tool decorator transforms this into a LangChain Tool object.
# See web_search.py for full explanation of @tool.

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Use for arithmetic, percentages,
    and unit conversions. Example inputs: '15 * 1.08', '(100 - 20) / 4'."""

    # EFFECT ON THE AGENT:
    #   Claude calls this when the user needs a precise calculation.
    #   The return value (a string like "7.17") becomes the ToolMessage content,
    #   which Claude reads before writing its final answer.

    try:
        # Step 1: Parse the expression string into an AST.
        # ast.parse("47.80 * 0.15", mode="eval") builds a tree structure.
        # mode="eval" means: parse as a single expression (not a full program).
        # If the expression has syntax errors, ast.parse raises SyntaxError here.
        tree = ast.parse(expression, mode="eval")

        # Step 2: Safely evaluate the AST tree using our whitelist walker.
        # tree.body = the root node of the expression tree (skips the Module wrapper).
        result = _safe_eval(tree.body)

        # Step 3: Convert the result to a string and return it.
        # str(result) turns float/int into a string: 7.17 → "7.17"
        # Tools must return strings — that's the contract with the agent loop.
        return str(result)

    except Exception as e:
        # PYTHON CONCEPT — f-string:
        #   f"Calculation error: {str(e)}"
        #   The f prefix enables expression interpolation inside {}.
        #   str(e) converts the exception to a readable message.
        #   In TypeScript: `Calculation error: ${String(e)}`
        #
        # Like web_search.py, returning a string instead of re-raising keeps
        # the agent loop running. Claude reads the error and handles it gracefully.
        return f"Calculation error: {str(e)}"
