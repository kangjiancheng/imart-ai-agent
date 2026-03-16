import { Chat, DateGroup, GroupedChats } from './types';

export function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

export function truncateTitle(text: string, maxLen = 40): string {
  const trimmed = text.trim();
  return trimmed.length <= maxLen ? trimmed : trimmed.slice(0, maxLen) + '…';
}

export function groupChatsByDate(chats: Chat[]): GroupedChats[] {
  const now = Date.now();
  const DAY = 86400000;

  const groups: Record<DateGroup, Chat[]> = {
    Today: [],
    Yesterday: [],
    'Last 7 days': [],
    Older: [],
  };

  for (const chat of chats) {
    const diff = now - chat.updatedAt;
    if (diff < DAY) {
      groups['Today'].push(chat);
    } else if (diff < 2 * DAY) {
      groups['Yesterday'].push(chat);
    } else if (diff < 7 * DAY) {
      groups['Last 7 days'].push(chat);
    } else {
      groups['Older'].push(chat);
    }
  }

  const order: DateGroup[] = ['Today', 'Yesterday', 'Last 7 days', 'Older'];
  return order
    .filter((g) => groups[g].length > 0)
    .map((g) => ({ group: g, chats: groups[g] }));
}
