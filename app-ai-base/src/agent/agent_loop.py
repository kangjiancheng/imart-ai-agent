import anthropic
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from src.llm.claude_client import llm, build_messages, build_system_prompt
from src.config.settings import settings
from src.tools.registry import TOOLS, tool_map
from src.memory.vector_memory import VectorMemory
from src.rag.retriever import RAGRetriever
from src.schemas.request import AgentRequest
from src.agent.token_budget import TokenBudget

MAX_ITERATIONS = 10


def _summarize_iterations(iterations: list[dict]) -> str:
    """Condense tool call history into a 1-2 sentence summary for memory storage."""
    if not iterations:
        return ""
    tool_names = [it["tool"] for it in iterations]
    tools_used = ", ".join(set(tool_names))
    count = len(iterations)
    return (
        f"Agent used {count} tool call(s): {tools_used}. "
        f"Last tool result snippet: {str(iterations[-1].get('result', ''))[:200]}"
    )


async def _extract_memory(user_message: str) -> str:
    """
    Ask Claude to extract a durable personal fact from a single user message.
    Returns a "User ..." sentence or "" if nothing is worth remembering.

    Uses a separate low-temperature ChatAnthropic instance (temperature=0,
    max_tokens=64, streaming=False) to produce a deterministic short output
    without the conversational flair of the main streaming llm.
    """
    extractor = ChatAnthropic(
        model=settings.claude_model,
        max_tokens=64,
        temperature=0,
        anthropic_api_key=settings.anthropic_api_key,
        anthropic_api_url=settings.anthropic_base_url,
        streaming=False,
    )

    extraction_prompt = (
        "Extract a durable personal fact from the message below.\n"
        "Output rules (strictly enforced):\n"
        "- If a personal fact exists (profession, skill, preference, goal, background): "
        "output EXACTLY one sentence. The sentence MUST start with the word 'User'. "
        "Example: 'User is a web developer learning AI agents with Python'\n"
        "- If no personal fact exists: output the single word NONE\n"
        "- Your entire response must be that one sentence or the word NONE. "
        "Do NOT write 'I understand', 'Got it', or anything else before or after."
    )

    response = await extractor.ainvoke([
        SystemMessage(content=extraction_prompt),
        HumanMessage(content=user_message),
    ])

    content = response.content
    if isinstance(content, list):
        result = "".join(
            block.get("text", "") for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
    else:
        result = content.strip()

    if result.upper() == "NONE":
        return ""
    if not result.startswith("User"):
        for sentence in result.split("."):
            sentence = sentence.strip()
            if sentence.startswith("User"):
                return sentence
        return ""
    return result


async def run(request: AgentRequest):
    """
    Main ReAct agent loop. Async generator — yields string tokens for SSE streaming.

    Flow:
      1. Recall long-term memory from Milvus for personalization.
      2. Build system prompt + message list (history + current message).
      3. Bind tools to planner (llm.bind_tools).
      4. ReAct loop (max MAX_ITERATIONS):
         - ainvoke → if tool_calls present, execute tools, append ToolMessages, repeat.
         - If no tool_calls → astream final answer, yield tokens, break.
      5. Extract personal facts and store session summary to long-term memory.
    """
    # ── Step 1: recall long-term memory ──────────────────────────────────────
    memory   = VectorMemory()
    recalled = await memory.recall(request.user_id, request.message, top_k=5)

    # ── Step 2: build initial message list ───────────────────────────────────
    system_prompt = build_system_prompt(
        recalled,
        document_context=request.document_context,
    )
    messages = build_messages(system_prompt, [m.model_dump() for m in request.history])
    messages.append(HumanMessage(content=request.message))

    # ── Step 3: set up planner and budget ────────────────────────────────────
    planner    = llm.bind_tools(TOOLS)
    budget     = TokenBudget()
    iterations = []
    tokens_used = 0

    # ── Step 4: ReAct loop ───────────────────────────────────────────────────
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
            # No more tools needed — stream the final answer.
            # Use planner.astream (not llm.astream) so tool schemas remain present;
            # bare llm rejects history containing tool_use/tool_result blocks.
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

        # ── Tool execution ────────────────────────────────────────────────────
        # Collect ALL tool results before appending to messages.
        # Anthropic API requires every tool_use block to have a matching
        # tool_result in the very next message — none can be missing.
        tool_results: list[tuple[dict, str]] = []

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            if tool_name == "rag_retrieve":
                # RAG is handled via RAGRetriever (async) rather than the stub in tool_map.
                retriever = RAGRetriever()
                docs      = await retriever.retrieve(tool_args.get("query", request.message))
                result    = retriever.format_for_prompt(docs)
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

        # Append assistant turn (with tool_calls) then all tool results.
        messages.append(response)
        for tool_call, result in tool_results:
            messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))

    # ── Step 5: write to long-term memory ────────────────────────────────────
    extracted = await _extract_memory(request.message)
    if extracted:
        await memory.store_if_new(request.user_id, extracted, tags=["user_fact"])

    if iterations:
        summary = _summarize_iterations(iterations)
        await memory.store_if_new(request.user_id, summary, tags=["session"])
