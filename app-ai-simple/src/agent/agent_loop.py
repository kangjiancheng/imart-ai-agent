import anthropic
from langchain_core.messages import HumanMessage, ToolMessage

from src.llm.claude_client import llm, build_messages, build_system_prompt
from src.tools.registry import TOOLS, tool_map
from src.memory.vector_memory import VectorMemory
from src.rag.retriever import RAGRetriever
from src.schemas.request import AgentRequest
from src.agent.token_budget import TokenBudget

MAX_ITERATIONS = 10


def _summarize_iterations(iterations: list[dict]) -> str:
    """Condense tool call history into a short summary for memory storage."""
    if not iterations:
        return ""
    tools_used = ", ".join(set(it["tool"] for it in iterations))
    count = len(iterations)
    return (
        f"Agent used {count} tool call(s): {tools_used}. "
        f"Last tool result snippet: {str(iterations[-1].get('result', ''))[:200]}"
    )


async def run(request: AgentRequest):
    """Main ReAct agent loop. Async generator — yields tokens for SSE streaming."""

    # Recall long-term memory to personalize the system prompt
    memory = VectorMemory()
    recalled = await memory.recall(request.user_id, request.message, top_k=5)

    system_prompt = build_system_prompt(
        recalled,
        document_context=request.document_context,
    )

    messages = build_messages(system_prompt, [m.model_dump() for m in request.history])
    messages.append(HumanMessage(content=request.message))

    planner = llm.bind_tools(TOOLS)
    budget = TokenBudget()
    iterations = []
    tokens_used = 0

    for i in range(MAX_ITERATIONS):
        messages = budget.trim_history(messages)

        try:
            response = await planner.ainvoke(messages)
        except anthropic.InternalServerError as exc:
            raise RuntimeError(
                "Anthropic service is temporarily unavailable (503). "
                "Please try again in a moment."
            ) from exc
        except anthropic.BadRequestError as exc:
            raise RuntimeError(
                "The conversation history sent to Claude was malformed "
                "(a tool_use block had no matching tool_result). "
                "Please start a new chat session."
            ) from exc

        tokens_used += response.usage_metadata.get("total_tokens", 0) if response.usage_metadata else 0
        budget.consume(tokens_used)

        if budget.is_exhausted or i >= MAX_ITERATIONS - 1:
            yield "[Agent reached context limit. Partial answer follows.] "
            break

        if not response.tool_calls:
            # No tool calls — stream the final answer
            # Use planner.astream (not llm.astream) to keep tool definitions present,
            # which is required when the history contains tool_use/tool_result blocks.
            async for chunk in planner.astream(messages):
                content = chunk.content
                if isinstance(content, str):
                    if content:
                        yield content
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "")
                            if text:
                                yield text
            break

        # Execute all requested tools and collect results
        tool_results: list[tuple[dict, str]] = []

        print(f"Iteration {i+1} tool calls:", response.tool_calls)
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            if tool_name == "rag_retrieve":
                # RAG is handled via RAGRetriever (async), not the tool_map stub
                retriever = RAGRetriever()
                docs = await retriever.retrieve(tool_args.get("query", request.message))
                result = retriever.format_for_prompt(docs)
            else:
                tool = tool_map.get(tool_name)
                if tool is None:
                    result = f"Error: unknown tool '{tool_name}'"
                else:
                    try:
                        result = await tool.ainvoke(tool_args)
                    except Exception as e:
                        result = f"Error: {tool_name} failed — {str(e)}"

            tool_results.append((tool_call, str(result)))
            iterations.append({"tool": tool_name, "args": tool_args, "result": str(result)})

        # Append the full response (with tool_calls) then each ToolMessage.
        # The Anthropic API requires this exact ordering — assistant turn before tool results.
        messages.append(response)
        for tool_call, result in tool_results:
            messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))

    # Write session summary to long-term memory after streaming completes
    if iterations:
        summary = _summarize_iterations(iterations)
        await memory.store_if_new(request.user_id, summary, tags=["session"])
