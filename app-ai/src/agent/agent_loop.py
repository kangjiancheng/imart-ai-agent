# src/agent/agent_loop.py
#
# ─────────────────────────────────────────────────────────────────────────────
# WHAT IS THIS FILE?
#
# This is the HEART of the entire AI agent. Every time a user sends a message,
# this function runs and decides what to do:
#   1. Should Claude answer directly?
#   2. Or should Claude call a tool first (search the web, run math, search docs)?
#
# This pattern is called "ReAct" = Reason → Act → Observe → repeat.
#
# WHY IS IT AN "async generator"?
#
# A normal function returns ONE value and stops.
# An async generator can `yield` MANY values over time — like a water tap dripping.
# We use this so the browser sees words appearing one-by-one as Claude types them,
# rather than waiting for the whole response before anything shows up.
#
# In Python:  `async def run() ... yield token`   ← this is an async generator
# In JS/TS:   like an AsyncGenerator or ReadableStream
# ─────────────────────────────────────────────────────────────────────────────

import anthropic
from langchain_core.messages import HumanMessage, ToolMessage
# HumanMessage  = a typed object representing one user message turn
#                 Claude needs typed objects, not plain {"role": "user", ...} dicts
# ToolMessage   = a typed object that carries a tool's result BACK to Claude
#                 It must include `tool_call_id` so Claude knows which call it answers

from src.llm.claude_client import llm, build_messages, build_system_prompt
# llm               = the ChatAnthropic singleton (one shared Claude connection)
# build_messages    = converts plain dicts → typed LangChain message objects
# build_system_prompt = assembles Claude's instructions + injected long-term memory

from src.tools.registry import TOOLS, tool_map
# TOOLS    = a list of all @tool functions Claude can call (web_search, calculator, etc.)
# tool_map = a dict {name → tool} so we can look up a tool by its name string

from src.memory.vector_memory import VectorMemory
# VectorMemory = reads/writes long-term user memory from/to Milvus (a vector database)
# "vector" = every piece of text is converted to a list of numbers (embedding)
#            that captures its *meaning*, not just exact words

from src.rag.retriever import RAGRetriever
# RAGRetriever = searches the knowledge base for documents relevant to the question
# RAG = Retrieval-Augmented Generation
#       Instead of only using Claude's training knowledge, we also search our own docs

from src.schemas.request import AgentRequest
# AgentRequest = the Pydantic model that validated the incoming HTTP request body
# Pydantic ≈ TypeScript's zod — validates and types data automatically

from src.agent.token_budget import TokenBudget
# TokenBudget = tracks how many tokens (words/subwords) we've used so far
# Claude has a 200,000-token limit per conversation — we must not exceed it

# Module-level constant — uppercase = convention for "never change this value"
MAX_ITERATIONS = 10
# The loop can run at most 10 times. Without this, a buggy tool could loop forever.


def _summarize_iterations(iterations: list[dict]) -> str:
    """
    After the loop finishes, condense what happened into 1-2 sentences.
    This summary gets saved to long-term memory so future sessions
    can recall what tools were used and what was found.

    iterations = a list like:
      [{"tool": "calculator", "args": {...}, "result": "51.0"}, ...]
    """
    if not iterations:
        return ""

    tool_names = [it["tool"] for it in iterations]
    # set() removes duplicates, so "calculator, calculator" becomes "calculator"
    tools_used = ", ".join(set(tool_names))
    count = len(iterations)
    return (
        f"Agent used {count} tool call(s): {tools_used}. "
        f"Last tool result snippet: {str(iterations[-1].get('result', ''))[:200]}"
        # [:200] = take only the first 200 characters — keep the summary short
    )


async def run(request: AgentRequest):
    """
    Main agent loop. An async generator — yields tokens for SSE streaming.
    Called directly by the FastAPI router (src/routers/agent.py).

    "async" = this function can pause and wait for I/O (network, DB) without
              blocking the whole server. Other requests can be served while waiting.

    "generator" = instead of returning once, it yields many values over time.
                  Each `yield` sends one piece of text to the browser immediately.
    """

    # ── Step 1: recall long-term memory ──────────────────────────────────────
    # We do this FIRST so we can inject what we know about this user
    # into the system prompt before Claude sees anything.
    memory   = VectorMemory()
    recalled = await memory.recall(request.user_id, request.message, top_k=5)
    # `await` = pause here and wait for the Milvus database to respond
    # top_k=5 = return the 5 most relevant memory chunks
    # recalled = a list of strings like ["User prefers bullet points", "Works in finance"]
    # If Milvus is down, recall() returns [] — the loop still works without personalization.

    # ── Step 2: build initial message list ───────────────────────────────────
    system_prompt = build_system_prompt(recalled)
    # system_prompt = a big string telling Claude who it is, what rules to follow,
    #                 and (optionally) what we know about this user from memory

    messages = build_messages(system_prompt, [m.model_dump() for m in request.history])
    # m.model_dump() = convert each Pydantic HistoryMessage object → plain dict
    # build_messages() then converts those dicts → typed LangChain message objects:
    #   [SystemMessage("You are a helpful..."), HumanMessage("Hello"), AIMessage("Hi!")]
    # This covers all PRIOR conversation turns (the context Claude needs to continue)

    messages.append(HumanMessage(content=request.message))
    # Now add the CURRENT user message at the end of the list.
    # Claude always reads the most recent message last — order matters!

    # ── Step 3: set up planner and budget ────────────────────────────────────
    planner = llm.bind_tools(TOOLS)
    # bind_tools() does TWO things:
    #   1. Tells Claude what tools exist (by embedding their schemas in the prompt)
    #   2. Configures Claude to respond with a structured "tool_call" instead of plain text
    #      when it wants to use a tool. No JSON parsing needed on our side.
    # Think of it like: llm + available tools = the "planner" that decides what to do

    budget     = TokenBudget()
    # Fresh budget for this request. Starts at 0 tokens used.

    iterations = []
    # A running log of every tool call this session — used to write memory at the end.
    # Each entry: {"tool": "calculator", "args": {...}, "result": "51.0"}

    tokens_used = 0
    # Running count of tokens consumed so far across all LLM calls in this loop.

    # ── Step 4: ReAct loop ───────────────────────────────────────────────────
    # This is the core "think → act → observe → repeat" cycle.
    # Each iteration = one round of asking Claude what to do next.
    for i in range(MAX_ITERATIONS):
        # i goes 0, 1, 2, ... 9. After 10 iterations we stop no matter what.

        # Trim history if token budget is running low
        messages = budget.trim_history(messages)
        # If the messages list has grown too large, drop the oldest pairs.
        # We always keep the SystemMessage — only old human/assistant turns are dropped.

        try:
            response = await planner.ainvoke(messages)
        except anthropic.InternalServerError as exc:
            # 503 = Anthropic is temporarily overloaded. Surface a clear message to the caller.
            raise RuntimeError(
                "Anthropic service is temporarily unavailable (503). "
                "Please try again in a moment."
            ) from exc

        # ainvoke() = "async invoke" — send ALL messages to Claude, wait for the FULL response.
        # We need the FULL response here (not streaming) because we have to READ
        # response.tool_calls to know whether Claude wants to call a tool.
        # If we streamed, tool_calls would be incomplete while tokens arrive.
        #
        # response is an AIMessage with two key attributes:
        #   response.content    = any reasoning text Claude wrote (may be empty)
        #   response.tool_calls = list of tool selections, e.g.:
        #     [{"name": "calculator", "args": {"expression": "15 * 1.08"}, "id": "tc_001"}]
        #     Empty list [] means Claude is ready to give the final answer.

        tokens_used += response.usage_metadata.get("total_tokens", 0) if response.usage_metadata else 0
        # usage_metadata = how many tokens this specific ainvoke() call consumed
        # We accumulate this into tokens_used so we can track total usage.
        budget.consume(tokens_used)
        # Tell the budget tracker how many tokens we've now used in total.

        if budget.is_exhausted or i >= MAX_ITERATIONS - 1:
            # Two ways to hit this:
            #   1. We've used nearly all 193,904 available tokens (context window full)
            #   2. We've looped 9 times already — this is iteration 9 (0-indexed)
            # Either way, gracefully tell the user we can't continue rather than crashing.
            yield "[Agent reached context limit. Partial answer follows.] "
            # `yield` = send this string to the browser RIGHT NOW as an SSE frame
            break
            # `break` = exit the for loop immediately

        if not response.tool_calls:
            # Empty tool_calls list means Claude is done reasoning.
            # It has enough information to write the final answer.
            # Now we STREAM the final answer token-by-token to the browser.
            #
            # IMPORTANT: use planner.astream (not llm.astream) so the same tool
            # definitions are present. Using bare llm caused the Anthropic API to
            # reject messages containing tool_use/tool_result history blocks.
            # Do NOT append `response` here — the conversation must end with a
            # user message (or ToolMessage). Appending AIMessage causes
            # "assistant message prefill" error on models that don't support it.
            async for chunk in planner.astream(messages):
                # When tools are bound, chunk.content can be either:
                #   - str: plain text (no tools used)
                #   - list: content blocks e.g. [{'type': 'text', 'text': '...', 'index': 0}]
                # We must handle both formats to extract the text correctly.
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
            # After streaming the full answer, exit the loop — we're done.

        # ── Tool execution ────────────────────────────────────────────────────
        # Claude wants to call a tool. Extract which tool and what arguments.
        tool_call = response.tool_calls[0]
        # We only process the first tool call per iteration (one at a time).
        # Multi-tool calling in parallel is a Phase 2 / LangGraph feature.

        tool_name = tool_call["name"]   # e.g. "calculator"
        tool_args = tool_call["args"]   # e.g. {"expression": "15 * 1.08"}

        if tool_name == "rag_retrieve":
            # RAG is a special case — we handle it via RAGRetriever, not the tool_map.
            # Why? Because RAGRetriever.retrieve() is async (awaitable), and the
            # @tool stub in registry.py is just a placeholder so Claude knows the name.
            retriever = RAGRetriever()
            docs      = await retriever.retrieve(tool_args.get("query", request.message))
            result    = retriever.format_for_prompt(docs)
            # result = a formatted string like:
            #   "[Document 1 — source: policy.pdf]\n... content ..."
        else:
            # Look up the tool in the tool_map dict by its name string.
            tool = tool_map.get(tool_name)
            if tool is None:
                # Claude hallucinated a tool name that doesn't exist.
                # Return an error observation — the LLM will see this and recover.
                result = f"Error: unknown tool '{tool_name}'"
            else:
                try:
                    result = await tool.ainvoke(tool_args)
                    # ainvoke() on a @tool = execute the tool function with the args dict.
                    # e.g. calculator.ainvoke({"expression": "15 * 1.08"}) → "16.2"
                except Exception as e:
                    result = f"Error: {tool_name} failed — {str(e)}"
                    # Tool errors become observations rather than crashing the loop.
                    # Claude sees the error message and decides what to do next
                    # (try a different approach, use a different tool, or give up).

        iterations.append({
            "tool":   tool_name,
            "args":   tool_args,
            "result": str(result),
        })
        # Log this iteration for the memory write step after streaming finishes.

        # Feed the result back so the planner sees it on the NEXT iteration.
        messages.append(response)
        # ↑ IMPORTANT: append the FULL Claude response (not just the text).
        #   The response object carries the tool_call inside it.
        #   The Anthropic API requires the assistant turn BEFORE the tool result.

        messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
        # ↑ ToolMessage = Claude's version of "here is the tool result".
        #   tool_call_id MUST match the "id" from the tool_call above.
        #   The Anthropic API validates this pairing — wrong id = 500 error.
        #   Think of it like a request/response pair: id links them together.
        #
        # After appending both, the messages list looks like:
        #   [..., HumanMessage("What is 15% of 340?"),
        #         AIMessage(tool_calls=[{name: calculator, ...}]),   ← just appended
        #         ToolMessage("51.0", tool_call_id="tc_001")]        ← just appended
        # On the next iteration, Claude sees all of this and can reason about the result.

    # ── Step 5: write to long-term memory ────────────────────────────────────
    # This runs AFTER the streaming is done. We never write memory DURING the loop
    # because Milvus writes add latency — the user would see a pause in the token stream.
    if iterations:
        summary = _summarize_iterations(iterations)
        # e.g. "Agent used 1 tool call(s): calculator. Last tool result snippet: 51.0"
        await memory.store_if_new(request.user_id, summary, tags=["session"])
        # store_if_new() = write this summary to the user_memory Milvus collection.
        # Future sessions for this user can recall this summary and personalize responses.
        # Silently skips if Milvus is down — the session already completed successfully.
