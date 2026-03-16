"use client";

import React from "react";
import { Message } from "../../lib/types";
import MarkdownRenderer from "./MarkdownRenderer";

interface Props {
  message: Message;
  isStreaming?: boolean; // show blinking cursor on last assistant message
}

export default function MessageBubble({ message, isStreaming }: Props) {
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <div className="flex justify-end mb-4">
        <div className="max-w-xl rounded-2xl rounded-tr-sm bg-zinc-200 dark:bg-zinc-700 text-zinc-900 dark:text-zinc-100 px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap break-words">
          {message.content}
        </div>
      </div>
    );
  }

  // Assistant message — render markdown
  return (
    <div className="flex justify-start mb-4">
      <div className="w-full rounded-2xl rounded-tl-sm bg-white dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100 px-4 py-3 text-sm leading-relaxed break-words">
        <MarkdownRenderer content={message.content} />
        {isStreaming && <span className="cursor-blink ml-0.5">▋</span>}
      </div>
    </div>
  );
}
