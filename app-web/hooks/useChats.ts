'use client';

import { useState, useCallback, useEffect } from 'react';
import { Chat, Message, Role } from '../lib/types';
import { loadChats, saveChats } from '../lib/storage';
import { generateId, truncateTitle } from '../lib/utils';

export function useChats() {
  const [chats, setChats] = useState<Chat[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);

  // Load from localStorage on mount
  useEffect(() => {
    const loaded = loadChats();
    setChats(loaded);
  }, []);

  // Persist to localStorage whenever chats change
  useEffect(() => {
    saveChats(chats);
  }, [chats]);

  const activeChat = chats.find((c) => c.id === activeChatId) ?? null;

  const createChat = useCallback((firstUserMessage: string): Chat => {
    const now = Date.now();
    const newChat: Chat = {
      id: generateId(),
      title: truncateTitle(firstUserMessage),
      messages: [],
      createdAt: now,
      updatedAt: now,
    };
    setChats((prev) => [newChat, ...prev]);
    setActiveChatId(newChat.id);
    return newChat;
  }, []);

  const appendMessage = useCallback(
    (chatId: string, role: Role, content: string): Message => {
      const msg: Message = {
        id: generateId(),
        role,
        content,
        createdAt: Date.now(),
      };
      setChats((prev) =>
        prev.map((c) =>
          c.id === chatId
            ? { ...c, messages: [...c.messages, msg], updatedAt: Date.now() }
            : c
        )
      );
      return msg;
    },
    []
  );

  // Update the last assistant message content (used during streaming)
  const updateLastAssistantMessage = useCallback(
    (chatId: string, content: string) => {
      setChats((prev) =>
        prev.map((c) => {
          if (c.id !== chatId) return c;
          const messages = [...c.messages];
          // Find last assistant message and update it
          for (let i = messages.length - 1; i >= 0; i--) {
            if (messages[i].role === 'assistant') {
              messages[i] = { ...messages[i], content };
              break;
            }
          }
          return { ...c, messages, updatedAt: Date.now() };
        })
      );
    },
    []
  );

  const selectChat = useCallback((id: string) => {
    setActiveChatId(id);
  }, []);

  const startNewChat = useCallback(() => {
    setActiveChatId(null);
  }, []);

  const deleteChat = useCallback(
    (id: string) => {
      setChats((prev) => prev.filter((c) => c.id !== id));
      if (activeChatId === id) setActiveChatId(null);
    },
    [activeChatId]
  );

  return {
    chats,
    activeChat,
    activeChatId,
    createChat,
    appendMessage,
    updateLastAssistantMessage,
    selectChat,
    startNewChat,
    deleteChat,
  };
}
