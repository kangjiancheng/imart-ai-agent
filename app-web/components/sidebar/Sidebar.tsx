"use client";

import React from "react";
import { Chat } from "../../lib/types";
import NewChatButton from "./NewChatButton";
import ChatHistoryList from "./ChatHistoryList";

interface Props {
  chats: Chat[];
  activeChatId: string | null;
  onNewChat: () => void;
  onSelectChat: (id: string) => void;
  onDeleteChat: (id: string) => void;
}

export default function Sidebar({
  chats,
  activeChatId,
  onNewChat,
  onSelectChat,
  onDeleteChat,
}: Props) {
  return (
    <aside className="flex h-full w-[250px] flex-shrink-0 flex-col bg-zinc-100 dark:bg-zinc-900 border-r border-zinc-200 dark:border-zinc-700">
      {/* Site name */}
      <div className="flex items-center px-3 py-4 border-b border-zinc-200 dark:border-zinc-700">
        <span className="text-sm font-semibold text-zinc-800 dark:text-zinc-100">
          AI Chat
        </span>
      </div>

      {/* New Chat */}
      <div className="px-2 py-2">
        <NewChatButton onClick={onNewChat} />
      </div>

      {/* All Chats label */}
      <p className="px-3 pt-2 pb-1 text-xs font-semibold text-zinc-400 dark:text-zinc-500 uppercase tracking-wider">
        All Chats
      </p>

      {/* Chat history */}
      <ChatHistoryList
        chats={chats}
        activeChatId={activeChatId}
        onSelect={onSelectChat}
        onDelete={onDeleteChat}
      />
    </aside>
  );
}
