'use client';

import React from 'react';
import { Chat } from '../../lib/types';
import { groupChatsByDate } from '../../lib/utils';

interface Props {
  chats: Chat[];
  activeChatId: string | null;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
}

export default function ChatHistoryList({
  chats,
  activeChatId,
  onSelect,
  onDelete,
}: Props) {
  const groups = groupChatsByDate(chats);

  if (groups.length === 0) {
    return (
      <div className="px-3 py-4 text-xs text-zinc-400 dark:text-zinc-500">
        No conversations yet.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4 overflow-y-auto flex-1 px-2 py-2">
      {groups.map(({ group, chats: groupChats }) => (
        <div key={group}>
          <p className="px-2 mb-1 text-xs font-semibold text-zinc-400 dark:text-zinc-500 uppercase tracking-wider">
            {group}
          </p>
          <ul className="flex flex-col gap-0.5">
            {groupChats.map((chat) => (
              <li key={chat.id} className="group relative">
                <button
                  onClick={() => onSelect(chat.id)}
                  className={`w-full text-left truncate rounded-lg px-3 py-2 text-sm transition-colors pr-8 ${
                    chat.id === activeChatId
                      ? 'bg-zinc-200 dark:bg-zinc-700 text-zinc-900 dark:text-zinc-100 font-medium'
                      : 'text-zinc-700 dark:text-zinc-300 hover:bg-zinc-200 dark:hover:bg-zinc-800'
                  }`}
                >
                  {chat.title}
                </button>
                {/* Delete button — visible on hover */}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onDelete(chat.id);
                  }}
                  className="absolute right-2 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 rounded p-0.5 hover:bg-zinc-300 dark:hover:bg-zinc-600 transition-opacity"
                  title="Delete chat"
                >
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="w-3.5 h-3.5 text-zinc-500 dark:text-zinc-400"
                  >
                    <polyline points="3 6 5 6 21 6" />
                    <path d="M19 6l-1 14H6L5 6" />
                    <path d="M10 11v6M14 11v6" />
                    <path d="M9 6V4h6v2" />
                  </svg>
                </button>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}
