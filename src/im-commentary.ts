/**
 * IM Commentary — sends human-readable tool-call explanations to IM channels
 * during long-running agent tasks. Uses Claude Haiku as a sidecar to translate
 * raw tool calls into natural language.
 *
 * Fire-and-forget: all public functions are async but callers should NOT await.
 * Rate-limited to prevent IM spam.
 */

import { logger } from './logger.js';

/** Minimum seconds between commentary messages per workspace folder. */
const RATE_LIMIT_SECONDS = 8;

/** Tools that are too trivial to comment on. */
const SKIP_TOOLS = new Set([
  'mcp__happyclaw__memory_query',
  'mcp__happyclaw__memory_remember',
  'TodoWrite',
  'TodoRead',
]);

/** Track last commentary time per folder to enforce rate limit. */
const lastCommentaryTime = new Map<string, number>();

/**
 * Generate and send a natural-language explanation of a tool call to IM.
 * Safe to call fire-and-forget — all errors are caught internally.
 */
export async function sendToolCommentary(opts: {
  folder: string;
  toolName: string;
  toolInputSummary?: string;
  isNested: boolean;
  sendMessage: (text: string) => Promise<void>;
}): Promise<void> {
  const { folder, toolName, toolInputSummary, isNested, sendMessage } = opts;

  // Only comment on top-level tool calls
  if (isNested) return;

  // Skip trivial tools
  if (SKIP_TOOLS.has(toolName)) return;

  // Rate limit: at most once per RATE_LIMIT_SECONDS per folder
  const now = Date.now();
  const last = lastCommentaryTime.get(folder) ?? 0;
  if (now - last < RATE_LIMIT_SECONDS * 1000) return;
  lastCommentaryTime.set(folder, now);

  try {
    const explanation = await generateExplanation(toolName, toolInputSummary);
    if (explanation) {
      await sendMessage(`🔧 ${explanation}`);
    }
  } catch (err) {
    logger.debug({ err, folder, toolName }, 'im-commentary: failed to send');
  }
}

/** Call Claude Haiku to generate a human-readable explanation. */
async function generateExplanation(
  toolName: string,
  inputSummary?: string,
): Promise<string | null> {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) return formatFallback(toolName, inputSummary);

  const input = inputSummary
    ? `工具: ${toolName}\n输入摘要: ${inputSummary.slice(0, 300)}`
    : `工具: ${toolName}`;

  const body = {
    model: 'claude-haiku-4-5-20251001',
    max_tokens: 60,
    messages: [
      {
        role: 'user',
        content: `用中文一句话（不超过20字）解释正在做什么：\n${input}\n\n只输出解释文字，不要标点以外的任何内容。`,
      },
    ],
  };

  const response = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(5000), // 5s timeout
  });

  if (!response.ok) {
    logger.debug({ status: response.status }, 'im-commentary: API error, using fallback');
    return formatFallback(toolName, inputSummary);
  }

  const data = (await response.json()) as {
    content: Array<{ type: string; text: string }>;
  };

  const text = data.content?.find((c) => c.type === 'text')?.text?.trim();
  return text || formatFallback(toolName, inputSummary);
}

/** Simple heuristic fallback when API is unavailable. */
function formatFallback(toolName: string, inputSummary?: string): string {
  const short = inputSummary ? inputSummary.slice(0, 60) : '';
  if (toolName === 'Bash') return short ? `执行命令: ${short}` : '执行 Shell 命令';
  if (toolName === 'Read') return short ? `读取文件: ${short}` : '读取文件';
  if (toolName === 'Write') return short ? `写入文件: ${short}` : '写入文件';
  if (toolName === 'Edit') return short ? `编辑文件: ${short}` : '编辑文件';
  if (toolName === 'Grep') return short ? `搜索: ${short}` : '搜索代码';
  if (toolName === 'Glob') return short ? `查找文件: ${short}` : '查找文件';
  if (toolName === 'Agent') return short ? `启动子代理: ${short}` : '启动子代理';
  if (toolName === 'WebFetch' || toolName === 'WebSearch') return short ? `网页: ${short}` : '访问网页';
  return short ? `${toolName}: ${short}` : toolName;
}
