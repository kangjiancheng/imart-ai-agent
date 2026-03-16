'use client';

import React, { useRef } from 'react';
import { useChats } from '../hooks/useChats';
import { useStream } from '../hooks/useStream';
import { AgentRequest, FileAttachment } from '../lib/types';
import { USER_ID } from '../lib/constants';
import Sidebar from './sidebar/Sidebar';
import ChatArea from './chat/ChatArea';

export default function ChatLayout() {
  const {
    chats,
    activeChat,
    activeChatId,
    createChat,
    appendMessage,
    updateLastAssistantMessage,
    selectChat,
    startNewChat,
    deleteChat,
  } = useChats();

  const { isStreaming, startStream, startFileStream, stopStream } = useStream();

  // Keep a ref to the chat ID being streamed into — survives re-renders
  const streamingChatIdRef = useRef<string | null>(null);
  // Accumulate streamed tokens before writing to state (avoids N re-renders)
  const streamBufferRef = useRef<string>('');

  // Shared callbacks wired up after we know the chatId
  const makeCallbacks = (chatId: string) => ({
    onToken: (token: string) => {
      streamBufferRef.current += token;
      updateLastAssistantMessage(chatId, streamBufferRef.current);
    },
    onDone: () => {
      streamingChatIdRef.current = null;
      streamBufferRef.current = '';
    },
    onError: (message: string) => {
      updateLastAssistantMessage(chatId, `⚠️ Error: ${message}`);
      streamingChatIdRef.current = null;
      streamBufferRef.current = '';
    },
  });

  // Called by ChatInput for both plain text and file+text submissions
  const handleSend = async (text: string, file?: FileAttachment) => {
    // ── 1. Resolve or create the chat ────────────────────────────────────────
    let chatId: string;
    let history: { role: 'user' | 'assistant'; content: string }[] = [];

    if (activeChat) {
      chatId = activeChat.id;
      history = activeChat.messages.map((m) => ({ role: m.role, content: m.content }));
    } else {
      const newChat = createChat(text);
      chatId = newChat.id;
    }

    streamingChatIdRef.current = chatId;
    streamBufferRef.current = '';

    // ── 2. Add the user message to the chat ──────────────────────────────────
    // Show the filename alongside the user's question so context is clear
    const userLabel = file ? `[📎 ${file.name}]\n${text}` : text;
    appendMessage(chatId, 'user', userLabel);

    // ── 3. Add an empty assistant placeholder ────────────────────────────────
    appendMessage(chatId, 'assistant', '');

    const callbacks = makeCallbacks(chatId);

    // ── 4a. File upload path → multipart POST to /v1/agent/chat-with-file ───
    if (file) {
      await startFileStream(
        { file: file.file, message: text, sessionId: chatId, history },
        callbacks
      );
      return;
    }

    // ── 4b. Plain text path → JSON POST to /v1/agent/chat ───────────────────
    const request: AgentRequest = {
      user_id: USER_ID,
      message: text,
      history,
      user_context: {
        subscription_tier: 'free',
        locale: navigator.language || 'en-US',
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
      },
      session_id: chatId,
      stream: true,
    };
    await startStream(request, callbacks);
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden">
      <Sidebar
        chats={chats}
        activeChatId={activeChatId}
        onNewChat={startNewChat}
        onSelectChat={selectChat}
        onDeleteChat={deleteChat}
      />
      <ChatArea
        activeChat={activeChat}
        isStreaming={isStreaming}
        onSend={handleSend}
        onStop={stopStream}
      />
    </div>
  );
}
