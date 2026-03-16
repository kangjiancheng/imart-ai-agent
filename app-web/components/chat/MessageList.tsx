"use client";

import React, { useEffect, useRef } from "react";
import { Message } from "../../lib/types";
import MessageBubble from "./MessageBubble";

interface Props {
  messages: Message[];
  isStreaming: boolean;
}

export default function MessageList({ messages, isStreaming }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new messages or while streaming
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isStreaming]);

  return (
    <div className="flex-1 overflow-y-auto px-4 py-6 scroll-smooth">
      <div className="mx-auto max-w-3xl">
        {messages.map((msg, idx) => {
          // Show streaming cursor only on the last assistant message while streaming
          const isLastAssistant =
            isStreaming &&
            msg.role === "assistant" &&
            idx === messages.length - 1;
          return (
            <MessageBubble
              key={msg.id}
              message={msg}
              isStreaming={isLastAssistant}
            />
          );
        })}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
