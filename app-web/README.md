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
- `react-markdown` — robust parser with full CommonMark + GFM support
- `remark-gfm` — tables, strikethrough, task lists
- `rehype-highlight` + `highlight.js` — syntax highlighting for 190+ languages with Atom One Dark theme
