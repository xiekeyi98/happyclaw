/**
 * TurnIndicator: shows current turn info and pending queue status.
 * Displayed above the StreamingDisplay when a turn is active.
 */
import { useChatStore } from '../../stores/chat';

interface TurnIndicatorProps {
  chatJid: string;
}

function formatChannel(channel: string): string {
  if (channel.startsWith('feishu:')) return '飞书';
  if (channel.startsWith('telegram:')) return 'Telegram';
  if (channel.startsWith('qq:')) return 'QQ';
  if (channel.startsWith('web:')) return 'Web';
  return channel;
}

function formatDuration(startedAt: number): string {
  const elapsed = Math.floor((Date.now() - startedAt) / 1000);
  if (elapsed < 60) return `${elapsed}s`;
  const min = Math.floor(elapsed / 60);
  const sec = elapsed % 60;
  return `${min}m${sec}s`;
}

export function TurnIndicator({ chatJid }: TurnIndicatorProps) {
  const activeTurn = useChatStore((s) => s.activeTurn[chatJid]);
  const pendingBuffer = useChatStore((s) => s.pendingBuffer[chatJid]);

  if (!activeTurn && !pendingBuffer) return null;

  const pendingEntries = pendingBuffer
    ? Object.entries(pendingBuffer).filter(([, count]) => count > 0)
    : [];

  if (!activeTurn && pendingEntries.length === 0) return null;

  return (
    <div className="flex items-center gap-2 px-3 py-1.5 text-xs text-gray-500 dark:text-gray-400 bg-gray-50 dark:bg-gray-800/50 rounded-md">
      {activeTurn && (
        <span className="flex items-center gap-1">
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-teal-500 animate-pulse" />
          <span>Turn</span>
          <span className="text-gray-400 dark:text-gray-500">·</span>
          <span>{formatChannel(activeTurn.channel)}</span>
          <span className="text-gray-400 dark:text-gray-500">·</span>
          <span>{activeTurn.messageCount} 条</span>
          <span className="text-gray-400 dark:text-gray-500">·</span>
          <span>{formatDuration(activeTurn.startedAt)}</span>
        </span>
      )}
      {pendingEntries.length > 0 && (
        <>
          {activeTurn && <span className="text-gray-300 dark:text-gray-600">|</span>}
          <span className="text-amber-600 dark:text-amber-400">
            {pendingEntries
              .map(([ch, count]) => `${formatChannel(ch)} ${count} 条等待中`)
              .join(' · ')}
          </span>
        </>
      )}
    </div>
  );
}
