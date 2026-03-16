'use client';

import React from 'react';
import { Chat, FileAttachment } from '../../lib/types';
import MessageList from './MessageList';
import WelcomeScreen from './WelcomeScreen';
import ChatInput from './ChatInput';

interface Props {
  activeChat: Chat | null;
  isStreaming: boolean;
  onSend: (text: string, file?: FileAttachment) => void;
  onStop: () => void;
}

export default function ChatArea({ activeChat, isStreaming, onSend, onStop }: Props) {
  return (
    <div className="flex flex-1 flex-col bg-white dark:bg-zinc-950 overflow-hidden">
      {activeChat ? (
        <MessageList messages={activeChat.messages} isStreaming={isStreaming} />
      ) : (
        <WelcomeScreen />
      )}
      <ChatInput onSend={onSend} isStreaming={isStreaming} onStop={onStop} />
    </div>
  );
}
