'use client';

import { useRef, useCallback, useState } from 'react';
import { AgentRequest, SSEEvent } from '../lib/types';
import { API_BASE_URL, USER_ID } from '../lib/constants';

interface StreamCallbacks {
  onToken: (token: string) => void;
  onDone: () => void;
  onError: (message: string) => void;
}

// Parameters for the /v1/agent/chat-with-file multipart endpoint
interface FileStreamParams {
  file: File;
  message: string;
  sessionId: string;
  history: { role: 'user' | 'assistant'; content: string }[];
}

export function useStream() {
  const [isStreaming, setIsStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  // Debug: gather all streamed data to inspect format
  const gatherStreamBufferData = useRef<string>('');

  // Shared SSE reader — both startStream and startFileStream use this
  const readSSE = useCallback(
    async (response: Response, callbacks: StreamCallbacks) => {
      if (!response.body) throw new Error('No response body');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      gatherStreamBufferData.current = ''; // Reset debug buffer

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        // SSE uses double newline as event delimiter
        const parts = buffer.split('\n\n');
        buffer = parts.pop()!;

        for (const part of parts) {
          const line = part.replace(/^data: /, '').trim();
          if (!line) continue;

          // Debug: log raw SSE event
          gatherStreamBufferData.current += line + '\n';
          console.log('[SSE Event]', line);

          try {
            const event: SSEEvent = JSON.parse(line);
            if (event.type === 'token') {
              // Unescape all JSON escape sequences
              // The backend sends: {"type":"token","content":"line1\\nline2"}
              // After JSON.parse, we get: {type:"token",content:"line1\nline2"} (actual newlines)
              // But sometimes there are double-escaped sequences, so we handle both
              let content = event.content;

              // Handle common escape sequences
              content = content
                .replace(/\\n/g, '\n')      // \n → newline
                .replace(/\\r/g, '\r')      // \r → carriage return
                .replace(/\\t/g, '\t')      // \t → tab
                .replace(/\\\\/g, '\\');    // \\ → single backslash

              console.log('[Token Unescaped]', { raw: event.content, unescaped: content });
              callbacks.onToken(content);
            }
            else if (event.type === 'done') {
              console.log('[Stream Complete] Full buffer:', gatherStreamBufferData.current);
              callbacks.onDone();
            }
            else if (event.type === 'error') callbacks.onError(event.message);
          } catch (err) {
            // Skip malformed SSE lines
            console.warn('[SSE Parse Error]', line, err);
          }
        }
      }
    },
    []
  );

  // POST /v1/agent/chat — JSON body
  const startStream = useCallback(
    async (request: AgentRequest, callbacks: StreamCallbacks) => {
      if (abortRef.current) abortRef.current.abort();
      abortRef.current = new AbortController();
      setIsStreaming(true);

      try {
        const response = await fetch(`${API_BASE_URL}/v1/agent/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(request),
          signal: abortRef.current.signal,
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        await readSSE(response, callbacks);
      } catch (err) {
        if (err instanceof Error && err.name === 'AbortError') return;
        callbacks.onError(err instanceof Error ? err.message : 'Unknown streaming error');
      } finally {
        setIsStreaming(false);
      }
    },
    [readSSE]
  );

  // POST /v1/agent/chat-with-file — multipart/form-data
  const startFileStream = useCallback(
    async (params: FileStreamParams, callbacks: StreamCallbacks) => {
      if (abortRef.current) abortRef.current.abort();
      abortRef.current = new AbortController();
      setIsStreaming(true);

      try {
        // Build multipart form — the backend reads each field by name
        const form = new FormData();
        form.append('file', params.file);
        form.append('message', params.message);
        form.append('user_id', USER_ID);
        form.append('session_id', params.sessionId);
        form.append('subscription_tier', 'free');
        form.append('locale', navigator.language || 'en-US');
        form.append('timezone', Intl.DateTimeFormat().resolvedOptions().timeZone);
        form.append('stream', 'true');
        // history must be sent as a JSON string (multipart can't carry nested objects)
        form.append('history_json', JSON.stringify(params.history));

        const response = await fetch(`${API_BASE_URL}/v1/agent/chat-with-file`, {
          method: 'POST',
          // Do NOT set Content-Type — browser sets it automatically with the boundary
          body: form,
          signal: abortRef.current.signal,
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        await readSSE(response, callbacks);
      } catch (err) {
        if (err instanceof Error && err.name === 'AbortError') return;
        callbacks.onError(err instanceof Error ? err.message : 'Unknown streaming error');
      } finally {
        setIsStreaming(false);
      }
    },
    [readSSE]
  );

  const stopStream = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    setIsStreaming(false);
  }, []);

  return { isStreaming, startStream, startFileStream, stopStream };
}
