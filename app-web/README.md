# app-web — iMart AI Agent Web UI

A ChatGPT-like web interface for the AI Agent Chat, built with Next.js 16 and Tailwind CSS 4. It connects to the FastAPI backend (`app-ai`) via SSE streaming and stores conversation history in localStorage.

---

## Features

- **Real-time streaming** — tokens appear character-by-character using `fetch`-based SSE (no `EventSource`)
- **File upload** — attach PDF, DOCX, TXT and other documents to any message; the file is sent to the AI agent for context-aware Q&A
- **Chat history** — conversations persisted in localStorage, grouped by Today / Yesterday / Last 7 days / Older
- **Markdown rendering** — assistant responses support headings, bold/italic, code blocks, lists, links
- **System dark/light mode** — respects OS preference via Tailwind `dark:` classes (no JS toggle needed)
- **Auto-resize input** — textarea grows with content up to 200px, then scrolls

---

## Prerequisites

| Tool           | Version                            |
| -------------- | ---------------------------------- |
| Node.js        | 18+                                |
| npm            | 9+                                 |
| app-ai backend | running at `http://localhost:8000` |

---

## Getting Started

```bash
# From the repo root
cd app-web

# Install dependencies
npm install

# Start development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

> The AI chat features require `app-ai` to be running. See `../app-ai/README.md` for setup instructions.

---

## Project Structure

```
app-web/
├── app/                        # Next.js App Router
│   ├── layout.tsx              # Root layout (fonts, metadata, html/body)
│   ├── page.tsx                # Entry point → renders <ChatLayout />
│   └── globals.css             # Global styles: scrollbar, blink cursor, .prose markdown
│
├── lib/                        # Pure utility modules (no React)
│   ├── types.ts                # TypeScript interfaces: Message, Chat, AgentRequest, SSEEvent, FileAttachment, ...
│   ├── constants.ts            # API_BASE_URL, STORAGE_KEY, USER_ID
│   ├── storage.ts              # loadChats() / saveChats() — localStorage helpers
│   └── utils.ts                # generateId(), truncateTitle(), groupChatsByDate()
│
├── hooks/                      # Custom React hooks
│   ├── useChats.ts             # Chat CRUD state with automatic localStorage sync
│   └── useStream.ts            # fetch-based SSE streaming with AbortController (JSON + multipart)
│
└── components/                 # React UI components
    ├── ChatLayout.tsx           # Root component — owns all state, orchestrates data flow
    ├── sidebar/
    │   ├── Sidebar.tsx          # Left panel shell (250px fixed width)
    │   ├── NewChatButton.tsx    # "New Chat" button with pencil icon
    │   └── ChatHistoryList.tsx  # Date-grouped chat list with delete on hover
    └── chat/
        ├── ChatArea.tsx         # Right panel — shows WelcomeScreen or active chat
        ├── WelcomeScreen.tsx    # Empty state displayed before first message
        ├── MessageList.tsx      # Scrollable message container with auto-scroll
        ├── MessageBubble.tsx    # Single message — user (right) or assistant (left)
        ├── MarkdownRenderer.tsx # Regex-based Markdown → HTML renderer (no library)
        └── ChatInput.tsx        # Auto-resize textarea + paperclip file attach + send/stop button
```

---

## Data Flow

```
ChatLayout  (useChats + useStream)
  │
  ├── Sidebar
  │     ├── NewChatButton  ──onNewChat──▶  startNewChat()
  │     └── ChatHistoryList ──onSelect──▶  selectChat(id)
  │
  └── ChatArea
        ├── MessageList    ◀── activeChat.messages
        └── ChatInput  ──onSend(text, file?)──▶  handleSend(text, file?)
                                                        │
                                            1. createChat() or use activeChat
                                            2. appendMessage(user)
                                            3. appendMessage(assistant, "")
                                                        │
                                         ┌──────────────┴──────────────┐
                                      file?                          no file
                                         │                              │
                               startFileStream()               startStream()
                               multipart POST                   JSON POST
                               /v1/agent/chat-with-file         /v1/agent/chat
                                         │                              │
                                         └──────────────┬──────────────┘
                                                        │
                                            onToken → updateLastAssistantMessage()
                                            onDone  → clear streaming state
                                            onError → write error into message
```

---

## localStorage Schema

```
Key:   "imart_chats"
Value: JSON.stringify(Chat[])   // newest-first array
```

Each `Chat` object:

```typescript
{
  id:        string,    // e.g. "1718000000000-abc1234"
  title:     string,    // first 40 chars of the opening message
  messages:  Message[],
  createdAt: number,    // Unix ms timestamp
  updatedAt: number,
}
```

---

## File Upload

The paperclip button in the chat input lets users attach a document to any message. The file is sent directly to the AI agent, which extracts the text and uses it as context when answering.

### How it works

1. User clicks the paperclip icon → browser file picker opens
2. Selected file appears as a preview chip above the input (name + size, with × to remove)
3. User types a question and presses Enter
4. Frontend sends a `multipart/form-data` POST to `/v1/agent/chat-with-file`
5. The backend extracts text from the file and injects it into Claude's context
6. Response streams back via SSE — same token format as regular chat

### Supported file types

| Extension                                          | Parser                               |
| -------------------------------------------------- | ------------------------------------ |
| `.pdf`                                             | PyMuPDF + Claude Vision OCR fallback |
| `.docx`                                            | python-docx (paragraphs + tables)    |
| `.txt` `.md` `.csv` `.json` `.yaml` `.html` `.xml` | UTF-8 decode                         |

### File size limit

The backend caps extracted text at **80,000 characters** (~20,000 tokens). Files that exceed this are truncated at the limit.

### Multipart form fields sent

| Field               | Value                            |
| ------------------- | -------------------------------- |
| `file`              | the binary file                  |
| `message`           | the user's question text         |
| `user_id`           | `USER_ID` constant               |
| `session_id`        | current chat ID                  |
| `subscription_tier` | `"free"`                         |
| `locale`            | `navigator.language`             |
| `timezone`          | `Intl.DateTimeFormat` timezone   |
| `stream`            | `"true"`                         |
| `history_json`      | JSON-stringified message history |

> `history_json` is sent as a JSON string (not a nested object) because multipart forms cannot carry structured types natively.

---

## SSE Streaming Protocol

The frontend sends a `POST /v1/agent/chat` request and reads the response body as a stream. Each SSE event is a JSON object on a `data:` line, terminated by `\n\n`:

```
data: {"type":"token","content":"Hello"}

data: {"type":"token","content":" world"}

data: {"type":"done"}
```

Error event: `{"type":"error","message":"..."}` — displayed inline in the assistant bubble.

---

## Color Palette

| Element                  | Light                    | Dark               |
| ------------------------ | ------------------------ | ------------------ |
| Sidebar background       | `bg-zinc-100`            | `dark:bg-zinc-900` |
| Chat area background     | `bg-white`               | `dark:bg-zinc-950` |
| User message bubble      | `bg-zinc-800 text-white` | same               |
| Assistant message bubble | `bg-zinc-100`            | `dark:bg-zinc-800` |
| Active chat item         | `bg-zinc-200`            | `dark:bg-zinc-700` |

---

## Available Scripts

| Command         | Description                        |
| --------------- | ---------------------------------- |
| `npm run dev`   | Start dev server at localhost:3000 |
| `npm run build` | Create production build            |
| `npm run start` | Serve production build             |
| `npm run lint`  | Run ESLint                         |

---

## Tech Stack

| Library      | Version | Purpose                      |
| ------------ | ------- | ---------------------------- |
| Next.js      | 16      | React framework (App Router) |
| React        | 19      | UI rendering                 |
| Tailwind CSS | 4       | Utility-first styling        |
| TypeScript   | 5       | Type safety                  |
| react-markdown | latest | Markdown rendering           |
| remark-gfm   | latest  | GitHub Flavored Markdown     |
| rehype-highlight | latest | Syntax highlighting for code |
| highlight.js | latest  | Code syntax colors           |

**Key markdown features:**
- `markdown-it` — robust parser with full CommonMark + GFM support
- Syntax highlighting for 190+ languages with Atom One Dark theme
- Proper handling of escaped newlines from backend SSE stream

---

## Developer Tips

### Debugging SSE Stream Data

The frontend includes built-in debugging for SSE events. Open the browser DevTools Console (F12) and look for:

- `[SSE Event]` — each raw SSE event received from the backend
- `[Token Unescaped]` — shows before/after unescaping of escape sequences
- `[Stream Complete]` — full accumulated buffer when streaming finishes
- `[SSE Parse Error]` — any JSON parsing errors

**Why this matters:** The backend sends markdown content with escaped newlines (`\n` as literal characters in JSON). The frontend unescapes these to actual newlines so `markdown-it` can properly parse headings, code blocks, and lists.

#### Complete Demo: Stream Data Handling

Here's the full flow with actual data:

**1. Backend sends SSE events (raw wire format):**
```
data: {"type":"token","content":"Here's a markdown"}

data: {"type":"token","content":" tutorial:\n\n```markdown"}

data: {"type":"token","content":"\n# Building an AI Agent"}

data: {"type":"done"}
```

Notice: newlines are escaped as `\n` in the JSON string (this is correct JSON encoding).

**2. Frontend receives and parses (in `useStream.ts`):**
```typescript
// Raw line from SSE stream
const line = 'data: {"type":"token","content":"Here\'s a markdown tutorial:\\n\\n```markdown"}';

// Step 1: Remove "data: " prefix and trim
const cleanLine = line.replace(/^data: /, '').trim();
// Result: {"type":"token","content":"Here's a markdown tutorial:\n\n```markdown"}

// Step 2: Parse JSON
const event = JSON.parse(cleanLine);
// Result: { type: "token", content: "Here's a markdown tutorial:\n\n```markdown" }
// Note: JSON.parse() automatically converts \n to actual newlines!

// Step 3: Unescape any remaining escape sequences
let content = event.content;
content = content
  .replace(/\\n/g, '\n')      // \n → newline
  .replace(/\\r/g, '\r')      // \r → carriage return
  .replace(/\\t/g, '\t')      // \t → tab
  .replace(/\\\\/g, '\\');    // \\ → single backslash

// Result: "Here's a markdown tutorial:\n\n```markdown"
// (with actual newlines, not escaped)

// Step 4: Pass to callback
callbacks.onToken(content);
```

**3. Frontend accumulates tokens in state:**
```typescript
// In ChatLayout.tsx
streamBufferRef.current += token;
// After all tokens: "Here's a markdown tutorial:\n\n```markdown\n# Building an AI Agent..."
```

**4. Frontend renders with markdown-it:**
```typescript
// In MarkdownRenderer.tsx
const html = md.render(streamBufferRef.current);
// markdown-it sees actual newlines and parses:
// - "# Building an AI Agent" → <h1>Building an AI Agent</h1>
// - "```markdown" → <pre><code>...</code></pre>
// - etc.
```

**5. Browser console output (for debugging):**
```
[SSE Event] {"type":"token","content":"Here's a markdown"}
[Token Unescaped] {
  raw: "Here's a markdown",
  unescaped: "Here's a markdown"
}
[SSE Event] {"type":"token","content":" tutorial:\n\n```markdown"}
[Token Unescaped] {
  raw: " tutorial:\n\n```markdown",
  unescaped: " tutorial:\n\n```markdown"
}
[SSE Event] {"type":"token","content":"\n# Building an AI Agent"}
[Token Unescaped] {
  raw: "\n# Building an AI Agent",
  unescaped: "\n# Building an AI Agent"
}
[SSE Event] {"type":"done"}
[Stream Complete] Full buffer: Here's a markdown tutorial:

```markdown
# Building an AI Agent...
```

#### Common Pitfalls

**Pitfall 1: Not unescaping before markdown parsing**
```typescript
// ❌ WRONG: Pass escaped content directly to markdown-it
md.render(event.content);  // markdown-it sees literal "\n" characters

// ✅ CORRECT: Unescape first
const unescaped = event.content.replace(/\\n/g, '\n');
md.render(unescaped);  // markdown-it sees actual newlines
```

**Pitfall 2: Forgetting JSON.parse() already unescapes**
```typescript
// ❌ WRONG: Double-unescaping
const line = '{"content":"line1\\nline2"}';
const event = JSON.parse(line);  // event.content = "line1\nline2" (actual newline)
const unescaped = event.content.replace(/\\n/g, '\n');  // No change needed!

// ✅ CORRECT: JSON.parse() handles it
const line = '{"content":"line1\\nline2"}';
const event = JSON.parse(line);  // event.content = "line1\nline2" (actual newline)
// Use event.content directly
```

**Pitfall 3: Splitting on newlines before unescaping**
```typescript
// ❌ WRONG: Split before unescaping
const lines = event.content.split('\n');  // Splits on literal "\n" strings, not newlines

// ✅ CORRECT: Unescape first, then split
const unescaped = event.content.replace(/\\n/g, '\n');
const lines = unescaped.split('\n');  // Splits on actual newlines
```

**Pitfall 4: Not handling all escape sequences**
```typescript
// ❌ WRONG: Only handle \n
content = content.replace(/\\n/g, '\n');

// ✅ CORRECT: Handle all common escapes
content = content
  .replace(/\\n/g, '\n')      // newline
  .replace(/\\r/g, '\r')      // carriage return
  .replace(/\\t/g, '\t')      // tab
  .replace(/\\\\/g, '\\')     // backslash
  .replace(/\\"/g, '"')       // quote
  .replace(/\\'/g, "'");      // apostrophe
```

### Markdown Rendering Pipeline

1. Backend sends SSE events with JSON-encoded content (newlines escaped)
2. Frontend receives and parses JSON
3. Frontend unescapes: `\n` → actual newline, `\t` → tab, etc.
4. `markdown-it` parses the unescaped markdown
5. Tailwind arbitrary selectors apply styling to HTML elements

If markdown doesn't render correctly:
1. Check browser console for `[Token Unescaped]` logs
2. Verify the unescaped content has actual newlines (not `\n` strings)
3. Ensure `markdown-it` is receiving valid markdown syntax

### File Upload Flow

1. User selects file via paperclip button
2. File preview chip appears above input
3. User types question + presses Enter
4. Frontend sends multipart POST to `/v1/agent/chat-with-file`
5. Backend extracts text from file (PDF/DOCX/TXT)
6. Backend injects extracted text into Claude's context
7. Response streams back as SSE (same format as regular chat)

**Supported formats:** PDF (with OCR fallback), DOCX, TXT, CSV, JSON, YAML, HTML, XML

### localStorage Schema

Chat data is stored as JSON in `localStorage` under key `"imart_chats"`:

```json
[
  {
    "id": "1718000000000-abc1234",
    "title": "First 40 chars of opening message",
    "messages": [
      { "role": "user", "content": "..." },
      { "role": "assistant", "content": "..." }
    ],
    "createdAt": 1718000000000,
    "updatedAt": 1718000000000
  }
]
```

To inspect: Open DevTools → Application → Local Storage → find `imart_chats`
