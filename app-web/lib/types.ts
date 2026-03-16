export type Role = 'user' | 'assistant';

export interface Message {
  id: string;
  role: Role;
  content: string;
  createdAt: number;
}

export interface Chat {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
}

export interface AgentRequest {
  user_id: string;
  message: string;
  history: { role: Role; content: string }[];
  user_context: {
    subscription_tier: 'free';
    locale: string;
    timezone: string;
  };
  session_id: string;
  stream: true;
}

export type SSEEvent =
  | { type: 'token'; content: string }
  | { type: 'done' }
  | { type: 'error'; message: string };

// Represents a file attached by the user before sending a message
export interface FileAttachment {
  file: File;       // the raw browser File object
  name: string;     // display name (file.name)
  sizeKB: number;   // rounded KB for display
}

export type DateGroup = 'Today' | 'Yesterday' | 'Last 7 days' | 'Older';

export interface GroupedChats {
  group: DateGroup;
  chats: Chat[];
}
