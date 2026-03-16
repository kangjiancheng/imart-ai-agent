'use client';

import React, { useRef, useCallback, useState, KeyboardEvent } from 'react';
import { FileAttachment } from '../../lib/types';

// Accepted MIME types — mirrors what app-ai's file_parser supports
const ACCEPTED = '.pdf,.docx,.txt,.md,.csv,.json,.yaml,.html,.xml';

interface Props {
  onSend: (text: string, file?: FileAttachment) => void;
  isStreaming: boolean;
  onStop: () => void;
}

export default function ChatInput({ onSend, isStreaming, onStop }: Props) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [attachment, setAttachment] = useState<FileAttachment | null>(null);

  const handleInput = () => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setAttachment({
      file: f,
      name: f.name,
      sizeKB: Math.round(f.size / 1024),
    });
    // Reset the input so the same file can be re-selected after removal
    e.target.value = '';
  };

  const removeAttachment = () => setAttachment(null);

  const submit = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    const text = el.value.trim();
    // Require at least some text (the backend needs a message field)
    if (!text || isStreaming) return;
    onSend(text, attachment ?? undefined);
    el.value = '';
    el.style.height = 'auto';
    setAttachment(null);
  }, [onSend, isStreaming, attachment]);

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="border-t border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-950 px-4 py-3">
      <div className="mx-auto max-w-2xl">

        {/* File preview chip — shown above the input box when a file is attached */}
        {attachment && (
          <div className="mb-2 flex items-center gap-2 rounded-lg border border-zinc-200 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-900 px-3 py-2 w-fit max-w-full">
            {/* File icon */}
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
              stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
              className="w-4 h-4 flex-shrink-0 text-zinc-500 dark:text-zinc-400">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
              <polyline points="14 2 14 8 20 8" />
            </svg>
            <span className="text-xs text-zinc-700 dark:text-zinc-300 truncate max-w-[220px]">
              {attachment.name}
            </span>
            <span className="text-xs text-zinc-400 dark:text-zinc-500 flex-shrink-0">
              {attachment.sizeKB} KB
            </span>
            {/* Remove button */}
            <button
              onClick={removeAttachment}
              className="flex-shrink-0 rounded p-0.5 hover:bg-zinc-200 dark:hover:bg-zinc-700 transition-colors"
              title="Remove file"
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
                stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"
                className="w-3 h-3 text-zinc-500">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </div>
        )}

        {/* Input row */}
        <div className="flex items-end gap-2 rounded-xl border border-zinc-300 dark:border-zinc-600 bg-white dark:bg-zinc-900 px-3 py-2 focus-within:ring-2 focus-within:ring-zinc-400 dark:focus-within:ring-zinc-500">

          {/* Hidden file input */}
          <input
            ref={fileInputRef}
            type="file"
            accept={ACCEPTED}
            className="hidden"
            onChange={handleFileChange}
          />

          {/* Paperclip / attach button */}
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isStreaming}
            className="flex-shrink-0 mb-1 rounded-lg p-1.5 text-zinc-400 dark:text-zinc-500 hover:text-zinc-600 dark:hover:text-zinc-300 hover:bg-zinc-100 dark:hover:bg-zinc-800 transition-colors disabled:opacity-40"
            title="Attach file (PDF, DOCX, TXT…)"
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
              stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
              className="w-4 h-4">
              <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66L9.41 17.41a2 2 0 0 1-2.83-2.83l8.49-8.48" />
            </svg>
          </button>

          <textarea
            ref={textareaRef}
            rows={1}
            placeholder={attachment ? 'Ask something about the file…' : 'Send a message…'}
            className="flex-1 resize-none bg-transparent text-sm text-zinc-900 dark:text-zinc-100 placeholder:text-zinc-400 dark:placeholder:text-zinc-500 outline-none py-1 max-h-[200px] overflow-y-auto"
            onInput={handleInput}
            onKeyDown={handleKeyDown}
            disabled={isStreaming}
          />

          {/* Stop or Send button */}
          {isStreaming ? (
            <button
              onClick={onStop}
              className="flex-shrink-0 mb-1 rounded-lg bg-zinc-200 dark:bg-zinc-700 hover:bg-zinc-300 dark:hover:bg-zinc-600 p-1.5 transition-colors"
              title="Stop streaming"
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"
                className="w-4 h-4 text-zinc-700 dark:text-zinc-200">
                <rect x="6" y="6" width="12" height="12" rx="2" />
              </svg>
            </button>
          ) : (
            <button
              onClick={submit}
              className="flex-shrink-0 mb-1 rounded-lg bg-zinc-800 dark:bg-zinc-200 hover:bg-zinc-700 dark:hover:bg-zinc-300 p-1.5 transition-colors disabled:opacity-40"
              title="Send message"
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
                stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
                className="w-4 h-4 text-white dark:text-zinc-900">
                <line x1="12" y1="19" x2="12" y2="5" />
                <polyline points="5 12 12 5 19 12" />
              </svg>
            </button>
          )}
        </div>

        <p className="mt-1.5 text-center text-xs text-zinc-400 dark:text-zinc-500">
          Press Enter to send · Shift+Enter for new line
          {attachment && ' · File will be sent with your message'}
        </p>
      </div>
    </div>
  );
}
