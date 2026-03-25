/**
 * Safety Hooks — Automatic risk detection and stuck-loop recovery.
 *
 * 1. PreToolUse: High-risk operation gatekeeper
 *    - Local rule engine scores risk before tool execution
 *    - High-risk operations get GPT "intent-action consistency" check
 *    - Can block dangerous operations or inject safety guidance
 *
 * 2. PostToolUse: Stuck/loop recovery coach
 *    - Tracks recent tool call fingerprints
 *    - Detects repetitive patterns (same command retried, same error recurring)
 *    - Calls GPT to suggest alternative approaches when stuck
 *
 * 3. PostToolUse: Invalid image detector
 *    - After Read tool returns, checks if the response contains image blocks
 *    - Validates image data via magic bytes (PNG/JPEG/GIF/WebP headers)
 *    - Warns agent if a file with image extension contains non-image data
 *    - Prevents corrupted "images" from poisoning the conversation history
 */

import type {
  HookCallback,
  PreToolUseHookInput,
  PostToolUseHookInput,
} from '@anthropic-ai/claude-agent-sdk';

import { callLlm, getLlmCredentials, hasLlmCredentials, type LlmCredentials } from './gpt-client.js';
import { detectImageMimeTypeFromBase64Strict } from './image-detector.js';
import {
  assessBashRisk,
  assessEditWriteRisk,
  redactToolInput,
  redactSecrets,
  type RiskAssessment,
} from './risk-rules.js';

// ─── Constants ─────────────────────────────────────────────

const GATEKEEPER_TIMEOUT_MS = 10_000;

// ─── Gatekeeper: GPT Check ────────────────────────────────

async function gptIntentCheck(
  creds: LlmCredentials,
  toolName: string,
  toolInput: unknown,
  risk: RiskAssessment,
): Promise<{ allow: boolean; guidance?: string }> {
  if (!hasLlmCredentials(creds)) return { allow: true };

  const inputStr = redactToolInput(toolName, toolInput);

  const prompt = `Agent 即将执行以下操作，请判断是否安全：

工具：${toolName}
输入：
${inputStr}

风险信号：
${risk.reasons.map(r => `- ${r}`).join('\n')}

请回答：
1. 该操作是否存在不可逆的破坏性风险？(是/否)
2. 是否有更安全的替代方案？(有：说明 / 无)
3. 结论：放行 / 需要确认 / 阻止

只回答以上三点，每点一行。`;

  try {
    const text = await callLlm(creds, {
      prompt,
      system: '你是安全审计员。简洁回答，不要废话。',
      model: 'gpt-5.4-mini',
      reasoningEffort: 'medium',
      timeoutMs: GATEKEEPER_TIMEOUT_MS,
    });

    const lower = text.toLowerCase();
    if (lower.includes('阻止') || lower.includes('block')) {
      return { allow: false, guidance: text };
    }
    if (lower.includes('需要确认') || lower.includes('confirm')) {
      return { allow: true, guidance: `⚠️ GPT 安全检查建议确认：\n${text}` };
    }
    return { allow: true };
  } catch {
    // GPT unavailable — degrade based on risk score
    if (risk.score >= 70) {
      // High risk + no GPT: require confirmation instead of silent pass
      return {
        allow: true,
        guidance: `⚠️ 高风险操作（分数 ${risk.score}），GPT 安全检查不可用，建议确认后执行。\n风险: ${risk.reasons.join('; ')}`,
      };
    }
    return { allow: true };
  }
}

// ─── Loop Detection ────────────────────────────────────────

interface ToolFingerprint {
  toolName: string;
  argsHash: string;
  exitCode?: number;
  timestamp: number;
  /** Redacted error snippet — safe to send externally */
  errorSnippet?: string;
}

const MAX_FINGERPRINTS = 15;
const LOOP_THRESHOLD = 3;
const ERROR_REPEAT_THRESHOLD = 2;
const COACH_COOLDOWN_MS = 60_000;

let recentFingerprints: ToolFingerprint[] = [];
let lastCoachTime = 0;

function hashArgs(toolName: string, toolInput: unknown): string {
  const inputStr = typeof toolInput === 'string'
    ? toolInput.slice(0, 200)
    : JSON.stringify(toolInput || '').slice(0, 200);
  let hash = 5381;
  const str = `${toolName}:${inputStr}`;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) + hash + str.charCodeAt(i)) & 0x7fffffff;
  }
  return hash.toString(36);
}

function extractErrorSnippet(toolResponse: unknown): string | undefined {
  const str = typeof toolResponse === 'string'
    ? toolResponse
    : JSON.stringify(toolResponse || '');
  const errorMatch = str.match(/(?:error|Error|ERROR|failed|FAILED|panic|PANIC)[^\n]{0,150}/);
  if (!errorMatch) return undefined;
  // Redact before storing — fingerprints may be sent to GPT
  return redactSecrets(errorMatch[0]);
}

function extractExitCode(toolResponse: unknown): number | undefined {
  const str = typeof toolResponse === 'string'
    ? toolResponse
    : JSON.stringify(toolResponse || '');
  const match = str.match(/exit code[:\s]+(\d+)/i);
  return match ? parseInt(match[1], 10) : undefined;
}

interface LoopDetection {
  isStuck: boolean;
  reason?: string;
  fingerprints?: ToolFingerprint[];
}

function detectLoop(newFp: ToolFingerprint): LoopDetection {
  recentFingerprints.push(newFp);
  if (recentFingerprints.length > MAX_FINGERPRINTS) {
    recentFingerprints = recentFingerprints.slice(-MAX_FINGERPRINTS);
  }

  // Check 1: Same command repeated N times
  const sameHash = recentFingerprints.filter(fp => fp.argsHash === newFp.argsHash);
  if (sameHash.length >= LOOP_THRESHOLD) {
    return {
      isStuck: true,
      reason: `同一命令已重复执行 ${sameHash.length} 次`,
      fingerprints: sameHash,
    };
  }

  // Check 2: Same error message repeated
  if (newFp.errorSnippet) {
    const sameError = recentFingerprints.filter(
      fp => fp.errorSnippet && fp.errorSnippet === newFp.errorSnippet
    );
    if (sameError.length >= ERROR_REPEAT_THRESHOLD) {
      return {
        isStuck: true,
        reason: `同一错误已出现 ${sameError.length} 次: ${newFp.errorSnippet?.slice(0, 80)}`,
        fingerprints: sameError,
      };
    }
  }

  // Check 3: High failure rate in recent calls
  const recent5 = recentFingerprints.slice(-5);
  const failures = recent5.filter(fp => fp.exitCode && fp.exitCode !== 0);
  if (recent5.length >= 5 && failures.length >= 4) {
    return {
      isStuck: true,
      reason: `最近 5 次调用中 ${failures.length} 次失败`,
      fingerprints: failures,
    };
  }

  return { isStuck: false };
}

async function getCoachAdvice(
  creds: LlmCredentials,
  detection: LoopDetection,
): Promise<string | null> {
  const now = Date.now();
  if (now - lastCoachTime < COACH_COOLDOWN_MS) return null;
  if (!hasLlmCredentials(creds)) return null;

  lastCoachTime = now;

  // Fingerprints are already redacted at collection time
  const recentCalls = recentFingerprints.slice(-8).map(fp =>
    `  ${fp.toolName} (${fp.exitCode !== undefined ? `exit=${fp.exitCode}` : 'ok'})${fp.errorSnippet ? `: ${fp.errorSnippet.slice(0, 80)}` : ''}`
  ).join('\n');

  const prompt = `Agent 似乎陷入了循环，需要换一个思路。

卡住原因：${detection.reason}

最近工具调用：
${recentCalls}

请给出 2-3 个具体的替代方案建议。要求：
- 每个建议一行
- 给出具体可执行的操作（不是"试试别的"这种空话）
- 如果看起来是环境/依赖问题，建议先诊断再修复`;

  try {
    return await callLlm(creds, {
      prompt,
      system: '你是开发顾问。给出具体的替代方案，简洁回复。',
      model: 'gpt-5.4-mini',
      reasoningEffort: 'medium',
      timeoutMs: GATEKEEPER_TIMEOUT_MS,
    });
  } catch {
    return null;
  }
}

// ─── Hook Factories ────────────────────────────────────────

/**
 * PreToolUse: High-risk operation gatekeeper.
 * Scores risk locally, calls GPT only for high-risk operations.
 */
export function createGatekeeperHook(log: (msg: string) => void): HookCallback {
  const creds = getLlmCredentials();

  return async (input, _toolUseId, _options) => {
    const hookInput = input as PreToolUseHookInput;
    const { tool_name, tool_input } = hookInput;

    // Skip subagent operations for non-Bash tools (Bash is too dangerous to skip)
    if (hookInput.agent_id && tool_name !== 'Bash') return {};

    let risk: RiskAssessment | null = null;

    if (tool_name === 'Bash') {
      const command = (tool_input as any)?.command || '';
      risk = assessBashRisk(command);
    } else if (tool_name === 'Edit' || tool_name === 'Write') {
      const filePath = (tool_input as any)?.file_path || (tool_input as any)?.path || '';
      const content = (tool_input as any)?.new_string || (tool_input as any)?.content || '';
      risk = assessEditWriteRisk(filePath, content);
    }

    if (!risk || !risk.needsGptCheck) return {};

    log(`[gatekeeper] Risk score ${risk.score} for ${tool_name}: ${risk.reasons.join(', ')}`);

    const result = await gptIntentCheck(creds, tool_name, tool_input, risk);

    if (!result.allow) {
      log(`[gatekeeper] BLOCKED ${tool_name}: ${result.guidance?.slice(0, 100)}`);
      return {
        decision: 'block' as const,
        reason: result.guidance || '高风险操作被安全检查阻止',
      };
    }

    if (result.guidance) {
      log(`[gatekeeper] Advisory for ${tool_name}`);
      return {
        hookSpecificOutput: {
          hookEventName: 'PreToolUse' as const,
          additionalContext: result.guidance,
        },
      };
    }

    return {};
  };
}

/**
 * PostToolUse: Stuck/loop recovery coach.
 * Tracks tool call patterns and detects repetitive failures.
 */
export function createLoopRecoveryHook(log: (msg: string) => void): HookCallback {
  const creds = getLlmCredentials();

  return async (input, _toolUseId, _options) => {
    const hookInput = input as PostToolUseHookInput;

    // Skip subagent operations
    if (hookInput.agent_id) return {};

    // Only track tools that can indicate stuck behavior
    const trackable = ['Bash', 'Edit', 'Write', 'Grep', 'Glob', 'Read'];
    if (!trackable.includes(hookInput.tool_name)) return {};

    const fingerprint: ToolFingerprint = {
      toolName: hookInput.tool_name,
      argsHash: hashArgs(hookInput.tool_name, hookInput.tool_input),
      exitCode: extractExitCode(hookInput.tool_response),
      timestamp: Date.now(),
      errorSnippet: extractErrorSnippet(hookInput.tool_response),
    };

    const detection = detectLoop(fingerprint);

    if (!detection.isStuck) return {};

    log(`[loop-coach] Stuck detected: ${detection.reason}`);

    const advice = await getCoachAdvice(creds, detection);
    if (!advice) return {};

    log(`[loop-coach] Injecting recovery advice`);
    return {
      hookSpecificOutput: {
        hookEventName: 'PostToolUse' as const,
        additionalContext: `⚠️ **循环检测**: ${detection.reason}\n\nGPT 建议换个思路：\n${advice}`,
      },
    };
  };
}

// ─── Invalid Image Detector ────────────────────────────────────────

/**
 * PostToolUse: Invalid image detector.
 * After Read tool reads a file with image extension, validates the response
 * contains actual image data (via magic bytes). If the file is not a real image
 * (e.g. an API error response saved as .png), warns the agent to prevent
 * corrupted "images" from poisoning the conversation history.
 *
 * Background: Feishu image download API can return error JSON while curl -o
 * saves it as .png. SDK Read tool then embeds this as base64 "image" in
 * conversation. On session resume, Claude API rejects with "Could not process
 * image" and the session becomes permanently broken.
 */
export function createImageValidationHook(log: (msg: string) => void): HookCallback {
  const IMAGE_EXTENSIONS = /\.(png|jpe?g|gif|webp|bmp|tiff?|avif|svg|ico)$/i;

  return async (input, _toolUseId, _options) => {
    const hookInput = input as PostToolUseHookInput;

    // Only check Read tool
    if (hookInput.tool_name !== 'Read') return {};

    // Check if the file has an image extension
    const toolInput = hookInput.tool_input as { file_path?: string } | undefined;
    const filePath = toolInput?.file_path;
    if (!filePath || !IMAGE_EXTENSIONS.test(filePath)) return {};

    // Check if the response contains image blocks with invalid data
    const response = hookInput.tool_response;
    const invalidImages = findInvalidImageBlocks(response);

    if (invalidImages.length === 0) return {};

    const warning = [
      `⚠️ **假图片检测**: \`${filePath}\` 文件扩展名是图片格式，但内容不是有效图片数据。`,
      `检测到 ${invalidImages.length} 个无效 image block。`,
      `这些假图片会污染会话历史，导致 session resume 时报 "Could not process image" 错误。`,
      ``,
      `**建议**：`,
      `1. 不要 Read 这个文件（它可能是 API 错误响应被错误保存为图片）`,
      `2. 用 \`file ${filePath}\` 命令检查文件实际类型`,
      `3. 如果是下载失败的图片，重新下载并验证 Content-Type`,
    ].join('\n');

    log(`[image-validator] Invalid image detected in ${filePath}: ${invalidImages.length} block(s)`);

    return {
      hookSpecificOutput: {
        hookEventName: 'PostToolUse' as const,
        additionalContext: warning,
      },
    };
  };
}

/**
 * Scan a tool_response value for image blocks with invalid base64 data.
 * The response structure from Read can be nested; we do a recursive search.
 */
function findInvalidImageBlocks(obj: unknown): string[] {
  const invalid: string[] = [];

  function walk(node: unknown): void {
    if (!node || typeof node !== 'object') return;

    if (Array.isArray(node)) {
      for (const item of node) walk(item);
      return;
    }

    const record = node as Record<string, unknown>;

    // Check if this is an image block with base64 source
    if (
      record.type === 'image' &&
      typeof record.source === 'object' &&
      record.source !== null
    ) {
      const source = record.source as Record<string, unknown>;
      if (source.type === 'base64' && typeof source.data === 'string') {
        const detected = detectImageMimeTypeFromBase64Strict(source.data);
        if (!detected) {
          // Base64 data doesn't match any known image format
          const preview = Buffer.from(
            (source.data as string).slice(0, 40),
            'base64',
          ).toString('utf-8').slice(0, 30);
          invalid.push(`media_type=${String(source.media_type)}, preview="${preview}..."`);
        }
      }
    }

    // Recurse into all values
    for (const value of Object.values(record)) {
      walk(value);
    }
  }

  walk(obj);
  return invalid;
}
