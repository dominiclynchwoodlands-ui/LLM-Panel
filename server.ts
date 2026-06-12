#!/usr/bin/env bun

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  ListToolsRequestSchema,
  CallToolRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import OpenAI from "openai";
import {
  PROVIDERS,
  isAvailable,
  listProviders,
  callProvider,
  getProvider,
  resolveApiKey,
  resolveBaseURL,
  unavailableReason,
  envPrefix,
} from "./providers.js";

// --- Configuration ---

// Limits that are global (not provider-specific)
const SESSION_IDLE_EXPIRY_MS = 30 * 60 * 1000; // 30 minutes
const MAX_SESSIONS = 10;
const MAX_TURNS_PER_SESSION = 50;

// Per-provider limits (maxInputChars, maxSessionChars) are read from the Provider
// record for every call — there is NO single global cap that all providers share.

// --- Startup availability check ---

const availableProviders = PROVIDERS.filter(isAvailable);
if (availableProviders.length === 0) {
  const statuses = listProviders()
    .map((s) => `  ${s.id} (${s.label}): ${s.reason ?? "unavailable"}`)
    .join("\n");
  console.error(
    `LLM Panel: no providers are available. Configure at least one provider's API key.\n\n${statuses}`
  );
  process.exit(1);
}

console.error(
  `LLM Panel: ${availableProviders.length} provider(s) available: ` +
    availableProviders.map((p) => p.label).join(", ")
);

// --- Session Management ---

interface Session {
  id: string;
  providerId: string;
  model: string;
  systemPrompt: string | undefined;
  messages: OpenAI.Chat.ChatCompletionMessageParam[];
  createdAt: number;
  lastActiveAt: number;
  tokensIn: number;
  tokensOut: number;
  turnCount: number;
  inFlight: boolean;
}

const sessions = new Map<string, Session>();

function generateSessionId(providerId: string): string {
  return `panel-${providerId}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

function expireStaleSessions(): void {
  const now = Date.now();
  for (const [id, session] of sessions) {
    if (!session.inFlight && now - session.lastActiveAt > SESSION_IDLE_EXPIRY_MS) {
      sessions.delete(id);
    }
  }
}

// Run cleanup every 5 minutes — unref so it doesn't keep the process alive
setInterval(expireStaleSessions, 5 * 60 * 1000).unref();

// --- Helpers ---

function formatUsage(
  usage: OpenAI.Completions.CompletionUsage | undefined
): string {
  if (!usage) return "usage unknown";
  return `${usage.prompt_tokens} in / ${usage.completion_tokens} out`;
}

function inputTooLarge(text: string, maxChars: number): boolean {
  return text.length > maxChars;
}

function renderReply(r: {
  content: string;
  reasoning: string;
  finishReason: string | null;
  model: string;
}): string {
  let body = r.content;
  if (!body) {
    body = r.reasoning
      ? "_(Model returned reasoning only — no final answer. This usually means it hit the token cap mid-think.)_"
      : "No response returned.";
  }
  if (r.finishReason === "length") {
    body +=
      "\n\n⚠️ _Output truncated at the token cap. Narrow the request or raise the provider's MAX_OUTPUT_TOKENS env var._";
  }
  return body;
}

function errorResult(message: string) {
  return {
    content: [{ type: "text" as const, text: message }],
    isError: true,
  };
}

function textResult(text: string) {
  return {
    content: [{ type: "text" as const, text }],
  };
}

/**
 * Validate that a value is a non-empty string (after trimming).
 * Returns an errorResult to bubble up, or null if the value is valid.
 */
function requireString(
  value: unknown,
  fieldName: string
): ReturnType<typeof errorResult> | null {
  if (typeof value !== "string" || value.trim().length === 0) {
    return errorResult(`Missing or empty required parameter '${fieldName}'.`);
  }
  return null;
}

/** Validate that a provider id is known and available. Returns error result or null. */
function validateProvider(
  providerId: unknown
): { error: ReturnType<typeof errorResult> } | null {
  if (typeof providerId !== "string" || !providerId) {
    return {
      error: errorResult(
        `Missing required parameter 'provider'. Valid ids: ${PROVIDERS.map((p) => p.id).join(", ")}.`
      ),
    };
  }
  const p = getProvider(providerId);
  if (!p) {
    return {
      error: errorResult(
        `Unknown provider: '${providerId}'. Valid ids: ${PROVIDERS.map((p) => p.id).join(", ")}.`
      ),
    };
  }
  if (!isAvailable(p)) {
    const missingKey = !resolveApiKey(p) ? p.apiKeyEnv.join(" or ") : null;
    const missingBase = !resolveBaseURL(p) ? `${envPrefix(providerId)}_BASE_URL` : null;
    const missing = [missingKey && `API key (${missingKey})`, missingBase]
      .filter(Boolean)
      .join(", ");
    return {
      error: errorResult(
        `Provider '${providerId}' (${p.label}) is not available: missing ${missing}.`
      ),
    };
  }
  return null;
}

function extractError(err: unknown): string {
  // Guard: only access properties when err is a non-null object.
  // This function runs inside catch handlers — it must NEVER itself throw.
  const isObj = err !== null && typeof err === "object";
  const status = isObj ? (err as { status?: number }).status : undefined;
  const code = isObj ? (err as { code?: string }).code : undefined;
  const message = err instanceof Error ? err.message : String(err);
  return [status && `HTTP ${status}`, code && `code=${code}`, message]
    .filter(Boolean)
    .join(" | ");
}

// --- MCP Server ---

const mcp = new Server(
  { name: "llm-panel", version: "3.0.0" },
  { capabilities: { tools: {} } }
);

// --- Tool Definitions ---

const PROVIDER_IDS_DOC = `Valid provider ids: ${PROVIDERS.map((p) => p.id).join(", ")}. Use panel_providers to check which are currently available.`;

const TOOLS = [
  {
    name: "panel_chat",
    description:
      "Send a single message to an LLM provider and get a response. " +
      "Stateless — no conversation history. For multi-turn, use panel_session_start.",
    inputSchema: {
      type: "object" as const,
      properties: {
        provider: {
          type: "string",
          description: `Which LLM provider to use. ${PROVIDER_IDS_DOC}`,
        },
        prompt: {
          type: "string",
          description: "The message to send",
        },
        system: {
          type: "string",
          description: "Optional system prompt to set the model's role/behavior",
        },
        model: {
          type: "string",
          description: "Optional model override. Defaults to the provider's default model.",
        },
      },
      required: ["provider", "prompt"],
    },
  },
  {
    name: "panel_code_review",
    description:
      "Send code to an LLM provider for an independent code review. " +
      "Returns structured feedback on bugs, security issues, architecture concerns. " +
      "Single-shot — for multi-turn review dialogue, use panel_session_start + panel_session_message.",
    inputSchema: {
      type: "object" as const,
      properties: {
        provider: {
          type: "string",
          description: `Which LLM provider to review with. ${PROVIDER_IDS_DOC}`,
        },
        code: {
          type: "string",
          description:
            "The code to review — a diff, full file, or multiple files concatenated with headers",
        },
        context: {
          type: "string",
          description:
            "What this code does, what changed and why, what the reviewer should know. More context = better review.",
        },
        focus: {
          type: "string",
          enum: ["general", "security", "bugs", "performance", "architecture"],
          description: "Review focus area (default: general)",
        },
        model: {
          type: "string",
          description: "Optional model override.",
        },
      },
      required: ["provider", "code"],
    },
  },
  {
    name: "panel_session_start",
    description:
      "Start a multi-turn conversation session with an LLM provider. " +
      "Returns a session ID. Use panel_session_message to send messages. " +
      "Sessions expire after 30 minutes of inactivity.",
    inputSchema: {
      type: "object" as const,
      properties: {
        provider: {
          type: "string",
          description: `Which LLM provider to start a session with. ${PROVIDER_IDS_DOC}`,
        },
        system: {
          type: "string",
          description:
            "System prompt that defines the model's role for this session. " +
            "Set this once — it persists for the entire session.",
        },
        context: {
          type: "string",
          description:
            "Initial context to include as the first user message (e.g., code to review, " +
            "sprint contract to evaluate). The model will acknowledge receiving it.",
        },
        model: {
          type: "string",
          description: "Optional model override for this session.",
        },
      },
      required: ["provider"],
    },
  },
  {
    name: "panel_session_message",
    description:
      "Send a message to an existing LLM Panel session. " +
      "The model sees the full conversation history — it remembers everything said in this session. " +
      "Each call is billed based on the full accumulated context.",
    inputSchema: {
      type: "object" as const,
      properties: {
        session_id: {
          type: "string",
          description: "The session ID returned by panel_session_start",
        },
        message: {
          type: "string",
          description: "Your message",
        },
      },
      required: ["session_id", "message"],
    },
  },
  {
    name: "panel_session_end",
    description:
      "End an LLM Panel session and get a usage summary. " +
      "Frees memory. Returns total tokens used across all turns.",
    inputSchema: {
      type: "object" as const,
      properties: {
        session_id: {
          type: "string",
          description: "The session ID to close",
        },
      },
      required: ["session_id"],
    },
  },
  {
    name: "panel_sessions_list",
    description:
      "List all active LLM Panel sessions with their provider, status, turn count, and token usage.",
    inputSchema: {
      type: "object" as const,
      properties: {},
    },
  },
  {
    name: "panel_consult",
    description:
      "Fan-out: send the same prompt to multiple LLM providers concurrently and return all responses. " +
      "Useful for comparing perspectives or getting a consensus. " +
      "One provider failing (timeout/error/unavailable) never fails the whole call — its error is captured per-element.",
    inputSchema: {
      type: "object" as const,
      properties: {
        prompt: {
          type: "string",
          description: "The prompt to send to all providers",
        },
        providers: {
          type: "array",
          items: { type: "string" },
          description: `Optional list of provider ids to consult. Defaults to all available providers. ${PROVIDER_IDS_DOC}`,
        },
        system: {
          type: "string",
          description: "Optional system prompt applied to all providers",
        },
        model: {
          type: "string",
          description: "Optional model override applied to all providers (uses each provider's default if not set)",
        },
      },
      required: ["prompt"],
    },
  },
  {
    name: "panel_providers",
    description:
      "List all registered LLM providers, their availability, and the reason if unavailable. " +
      "Use this to discover which providers are wired and ready before calling other panel tools.",
    inputSchema: {
      type: "object" as const,
      properties: {},
    },
  },
];

mcp.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: TOOLS }));

// --- Code Review System Prompt & Focus ---

const CODE_REVIEW_SYSTEM = `You are an expert code reviewer providing an independent second opinion from a different AI model. Your review runs alongside Claude's analysis to catch blind spots that same-model review misses.

Rules:
- Be direct, specific, and actionable
- Reference exact code snippets or line numbers
- Do NOT pad with praise — the user wants issues and risks, not reassurance
- If you find nothing wrong, say so briefly — don't invent problems
- Flag anything that "smells off" even if you can't prove it's a bug

Structure your review as:

## Critical Issues
Bugs, security vulnerabilities, data loss risks — MUST fix before shipping.
(If none, write "None found.")

## Warnings
Performance problems, race conditions, edge cases, error handling gaps — SHOULD fix.
(If none, write "None found.")

## Observations
Architecture notes, maintainability suggestions, patterns worth reconsidering — nice to have.
(If none, omit this section.)

## Verdict
One line: PASS | WARN | FAIL — with a brief reason.`;

const FOCUS_INSTRUCTIONS: Record<string, string> = {
  general:
    "Perform a comprehensive review covering correctness, security, performance, and architecture.",
  security:
    "Focus specifically on security: injection vectors, auth bypass, data exposure, secrets handling, OWASP top 10. Other issues are secondary.",
  bugs:
    "Focus specifically on correctness: logic errors, off-by-ones, null/undefined handling, race conditions, edge cases, error propagation.",
  performance:
    "Focus specifically on performance: unnecessary allocations, O(n^2) patterns, blocking I/O in hot paths, memory leaks, cache misses.",
  architecture:
    "Focus specifically on architecture: coupling, abstraction boundaries, testability, separation of concerns, API surface design.",
};

// --- Tool Handlers ---

async function handleChat(args: Record<string, unknown>) {
  const providerId = args.provider;
  const check = validateProvider(providerId);
  if (check) return check.error;

  const promptErr = requireString(args.prompt, "prompt");
  if (promptErr) return promptErr;
  const prompt = args.prompt as string;
  const system = args.system as string | undefined;
  const model = args.model as string | undefined;
  const p = getProvider(providerId as string)!;

  if (inputTooLarge(prompt, p.maxInputChars)) {
    return errorResult(
      `Input too large: ${prompt.length} chars (max ${p.maxInputChars} for ${p.label}).`
    );
  }

  const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [];
  if (system) messages.push({ role: "system", content: system });
  messages.push({ role: "user", content: prompt });

  const result = await callProvider(providerId as string, messages, { model });
  const reply = renderReply(result);
  const usage = formatUsage(result.usage);

  return textResult(`${reply}\n\n---\n_${result.model} (${providerId}) | ${usage}_`);
}

async function handleCodeReview(args: Record<string, unknown>) {
  const providerId = args.provider;
  const check = validateProvider(providerId);
  if (check) return check.error;

  const codeErr = requireString(args.code, "code");
  if (codeErr) return codeErr;
  const code = args.code as string;
  const context = (args.context as string) || "";
  const focus = (args.focus as string) || "general";
  const model = args.model as string | undefined;

  const p = getProvider(providerId as string)!;

  if (inputTooLarge(code, p.maxInputChars)) {
    return errorResult(
      `Input too large: ${code.length} chars (max ${p.maxInputChars} for ${p.label}). Send a focused diff instead of full files.`
    );
  }

  const instruction = FOCUS_INSTRUCTIONS[focus] || FOCUS_INSTRUCTIONS.general;

  const userContent = [
    `**Review instruction:** ${instruction}`,
    context && `**Context:** ${context}`,
    "**Code to review:**",
    "```",
    code,
    "```",
  ]
    .filter(Boolean)
    .join("\n\n");

  const result = await callProvider(
    providerId as string,
    [
      { role: "system", content: CODE_REVIEW_SYSTEM },
      { role: "user", content: userContent },
    ],
    { model }
  );

  const review = renderReply(result);
  const usage = formatUsage(result.usage);

  return textResult(
    `# Independent Review — ${p.label} (${result.model})\n\n${review}\n\n---\n_${usage}_`
  );
}

async function handleSessionStart(args: Record<string, unknown>) {
  const providerId = args.provider;
  const check = validateProvider(providerId);
  if (check) return check.error;

  expireStaleSessions();

  if (sessions.size >= MAX_SESSIONS) {
    return errorResult(
      `Too many active sessions (max ${MAX_SESSIONS}). End an existing session first, or wait for idle expiry.`
    );
  }

  const system = args.system as string | undefined;
  const context = args.context as string | undefined;
  const modelOverride = args.model as string | undefined;
  const p = getProvider(providerId as string)!;
  const model = modelOverride ?? p.defaultModel;
  const id = generateSessionId(providerId as string);

  const session: Session = {
    id,
    providerId: providerId as string,
    model,
    systemPrompt: system,
    messages: [],
    createdAt: Date.now(),
    lastActiveAt: Date.now(),
    tokensIn: 0,
    tokensOut: 0,
    turnCount: 0,
    inFlight: false,
  };

  if (system) {
    session.messages.push({ role: "system", content: system });
  }

  // If initial context provided, send it as the first turn
  if (context) {
    if (inputTooLarge(context, p.maxInputChars)) {
      return errorResult(
        `Context too large: ${context.length} chars (max ${p.maxInputChars} for ${p.label}).`
      );
    }

    const messagesForApi = [...session.messages, { role: "user" as const, content: context }];

    // Reserve the slot BEFORE the awaited callProvider to close the TOCTOU window where
    // two concurrent context-bearing starts could both pass the MAX_SESSIONS check.
    // Mark inFlight so expireStaleSessions does not evict the reserved session during a
    // long initial call.
    session.inFlight = true;
    sessions.set(id, session);
    try {
      const result = await callProvider(providerId as string, messagesForApi, { model });
      const reply = renderReply(result);

      // Only persist messages/tokens to session after successful API call
      session.messages.push({ role: "user", content: context });
      session.messages.push({ role: "assistant", content: result.content || "(no answer returned)" });
      session.tokensIn += result.usage?.prompt_tokens ?? 0;
      session.tokensOut += result.usage?.completion_tokens ?? 0;
      session.turnCount = 1;
      session.lastActiveAt = Date.now();
      session.inFlight = false;

      return textResult(
        `**Session started: \`${id}\`** (${p.label})\n\nInitial response:\n\n${reply}\n\n---\n_Turn 1 | ${formatUsage(result.usage)} | Use panel_session_message to continue_`
      );
    } catch (err) {
      // Roll back the reservation — a failed initial call must not leave a phantom session.
      sessions.delete(id);
      throw err;
    }
  }

  sessions.set(id, session);
  return textResult(
    `**Session started: \`${id}\`** (${p.label})\n\nNo initial context sent — session is ready. Use \`panel_session_message\` with this session ID to begin the conversation.`
  );
}

async function handleSessionMessage(args: Record<string, unknown>) {
  const sessionId = args.session_id as string;
  const messageErr = requireString(args.message, "message");
  if (messageErr) return messageErr;
  const message = args.message as string;

  const session = sessions.get(sessionId);
  if (!session) {
    return errorResult(
      `Session not found: ${sessionId}. It may have expired (30 min idle limit) or been ended.`
    );
  }

  const sessionProvider = getProvider(session.providerId)!;

  if (inputTooLarge(message, sessionProvider.maxInputChars)) {
    return errorResult(
      `Message too large: ${message.length} chars (max ${sessionProvider.maxInputChars} for ${sessionProvider.label}).`
    );
  }

  if (session.turnCount >= MAX_TURNS_PER_SESSION) {
    return errorResult(
      `Session \`${sessionId}\` has reached the maximum of ${MAX_TURNS_PER_SESSION} turns. End this session and start a new one.`
    );
  }

  // Soft cap: char-based guard, NOT a token guard — actual token usage depends on the model's
  // tokenizer and is higher than raw char count. This prevents extreme runaway sessions;
  // it does NOT guarantee the request will fit in the model's context window exactly.
  const projectedChars =
    session.messages.reduce((n, m) => n + (typeof m.content === "string" ? m.content.length : 0), 0) +
    message.length;
  if (projectedChars > sessionProvider.maxSessionChars) {
    return errorResult(
      `Session \`${sessionId}\` context is too large (${projectedChars} chars > ${sessionProvider.maxSessionChars} for ${sessionProvider.label}). Start a fresh session — long histories eventually exceed the model context window.`
    );
  }

  if (session.inFlight) {
    return errorResult(
      `Session \`${sessionId}\` is already processing a message. Wait for it to finish before sending another.`
    );
  }

  session.inFlight = true;
  session.lastActiveAt = Date.now();

  try {
    // Build messages for API without mutating session state yet
    const messagesForApi = [...session.messages, { role: "user" as const, content: message }];

    const result = await callProvider(session.providerId, messagesForApi, { model: session.model });
    const reply = renderReply(result);

    // Only persist after successful API call — prevents corrupted state on failure
    session.messages.push({ role: "user", content: message });
    session.messages.push({ role: "assistant", content: result.content || "(no answer returned)" });
    session.tokensIn += result.usage?.prompt_tokens ?? 0;
    session.tokensOut += result.usage?.completion_tokens ?? 0;
    session.turnCount++;

    return textResult(
      `${reply}\n\n---\n_Session \`${sessionId}\` | Turn ${session.turnCount}/${MAX_TURNS_PER_SESSION} | This turn: ${formatUsage(result.usage)} | Cumulative: ${session.tokensIn} in / ${session.tokensOut} out_`
    );
  } finally {
    session.inFlight = false;
    session.lastActiveAt = Date.now();
  }
}

async function handleSessionEnd(args: Record<string, unknown>) {
  const sessionId = args.session_id as string;

  const session = sessions.get(sessionId);
  if (!session) {
    return errorResult(
      `Session not found: ${sessionId}. It may have already expired or been ended.`
    );
  }

  sessions.delete(sessionId);

  const durationMs = Date.now() - session.createdAt;
  const durationMin = Math.round(durationMs / 60_000);
  const p = getProvider(session.providerId);

  return textResult(
    `**Session \`${sessionId}\` ended.**\n\n` +
      `- Provider: ${p?.label ?? session.providerId}\n` +
      `- Model: ${session.model}\n` +
      `- Turns: ${session.turnCount}\n` +
      `- Duration: ${durationMin} min\n` +
      `- Total tokens: ${session.tokensIn} in / ${session.tokensOut} out\n` +
      `- Messages in history: ${session.messages.length}`
  );
}

async function handleSessionsList() {
  expireStaleSessions();

  if (sessions.size === 0) {
    return textResult("No active LLM Panel sessions.");
  }

  const lines: string[] = ["**Active LLM Panel Sessions:**\n"];
  for (const [id, s] of sessions) {
    const p = getProvider(s.providerId);
    const idleMin = Math.round((Date.now() - s.lastActiveAt) / 60_000);
    const expiresIn = Math.max(
      0,
      Math.round((SESSION_IDLE_EXPIRY_MS - (Date.now() - s.lastActiveAt)) / 60_000)
    );
    lines.push(
      `- \`${id}\` — ${p?.label ?? s.providerId} | ${s.model} | ${s.turnCount} turns, ${s.tokensIn + s.tokensOut} total tokens, idle ${idleMin}m, expires in ~${expiresIn}m`
    );
  }

  return textResult(lines.join("\n"));
}

async function handleConsult(args: Record<string, unknown>) {
  const promptErr = requireString(args.prompt, "prompt");
  if (promptErr) return promptErr;
  const prompt = args.prompt as string;
  const system = args.system as string | undefined;
  const model = args.model as string | undefined;
  const requestedIds = args.providers as string[] | undefined;

  // Resolve the full target list — never abort early on a bad/unavailable id.
  // Unknown or unavailable ids become pre-computed error elements; only valid+available
  // ids reach callProvider. The call as a whole NEVER returns errorResult due to a bad id.
  interface ConsultResult {
    provider: string;
    label: string;
    model: string;
    answer: string;
    usage: string;
    error?: string;
  }

  let allIds: string[];
  if (requestedIds !== undefined) {
    // Caller explicitly passed `providers` — validate it is a non-empty array of strings.
    // Short-circuit order matters: Array.isArray first so .some is never reached on a non-array.
    if (
      !Array.isArray(requestedIds) ||
      requestedIds.length === 0 ||
      requestedIds.some((id) => typeof id !== "string")
    ) {
      return errorResult(
        "`providers`, if provided, must be a non-empty array of provider id strings — omit it to consult all available providers."
      );
    }
    allIds = requestedIds;
  } else {
    allIds = PROVIDERS.filter(isAvailable).map((p) => p.id);
  }

  if (allIds.length === 0) {
    // No providers at all — return a structured single-element error, not a thrown abort.
    return textResult(
      "**Consult summary:** 0/0 providers responded.\n\nNo providers are available. " +
        "Configure at least one provider key and restart the server."
    );
  }

  // Classify each id upfront: pre-error (unknown/unavailable/oversized) or callable.
  type PreError = { kind: "pre-error"; id: string; label: string; reason: string };
  type Callable = { kind: "callable"; id: string };
  type Slot = PreError | Callable;

  const slots: Slot[] = allIds.map((id): Slot => {
    const p = getProvider(id);
    if (!p) {
      return { kind: "pre-error", id, label: id, reason: `unknown provider '${id}'` };
    }
    if (!isAvailable(p)) {
      const r = unavailableReason(p) ?? "unavailable";
      return { kind: "pre-error", id, label: p.label, reason: r };
    }
    if (inputTooLarge(prompt, p.maxInputChars)) {
      return {
        kind: "pre-error",
        id,
        label: p.label,
        reason: `prompt too large (${prompt.length} chars > ${p.maxInputChars} max)`,
      };
    }
    return { kind: "callable", id };
  });

  const callableSlots = slots.filter((s): s is Callable => s.kind === "callable");

  const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [];
  if (system) messages.push({ role: "system", content: system });
  messages.push({ role: "user", content: prompt });

  // Fan out only the callable providers — Promise.allSettled isolates each failure.
  const settled = await Promise.allSettled(
    callableSlots.map((s) => callProvider(s.id, messages, { model }))
  );

  // Map settled results back by index into callableSlots.
  const settledByCallableId = new Map<string, PromiseSettledResult<Awaited<ReturnType<typeof callProvider>>>>();
  callableSlots.forEach((s, i) => settledByCallableId.set(s.id, settled[i]));

  // Build one result element per original id, preserving input order.
  const results: ConsultResult[] = slots.map((slot) => {
    if (slot.kind === "pre-error") {
      return { provider: slot.id, label: slot.label, model: "—", answer: "", usage: "—", error: slot.reason };
    }
    const p = getProvider(slot.id)!;
    const outcome = settledByCallableId.get(slot.id)!;
    if (outcome.status === "fulfilled") {
      return {
        provider: slot.id,
        label: p.label,
        model: outcome.value.model,
        answer: renderReply(outcome.value),
        usage: formatUsage(outcome.value.usage),
      };
    } else {
      return {
        provider: slot.id,
        label: p.label,
        model: p.defaultModel,
        answer: "",
        usage: "—",
        error: extractError(outcome.reason),
      };
    }
  });

  const succeeded = results.filter((r) => !r.error);
  const failed = results.filter((r) => !!r.error);

  const summary =
    `**Consult summary:** ${succeeded.length}/${allIds.length} providers responded` +
    (failed.length > 0
      ? ` (failed: ${failed.map((r) => r.label).join(", ")})`
      : ".");

  const sections = results.map((r) => {
    if (r.error) {
      return `## ${r.label}\n\n**Error:** ${r.error}`;
    }
    return `## ${r.label} (${r.model})\n\n${r.answer}\n\n_${r.usage}_`;
  });

  return textResult([summary, "", ...sections].join("\n\n"));
}

async function handleProviders() {
  const statuses = listProviders();
  const lines = statuses.map((s) => {
    const avail = s.available ? "✓ available" : `✗ unavailable (${s.reason})`;
    return `- **${s.id}** (${s.label}): ${avail} | default model: \`${s.defaultModel}\`${s.baseURL ? ` | baseURL: ${s.baseURL}` : ""}`;
  });
  return textResult(
    `**LLM Panel — Registered Providers:**\n\n${lines.join("\n")}`
  );
}

// --- Request Router ---

mcp.setRequestHandler(CallToolRequestSchema, async (req) => {
  const { name, arguments: args } = req.params;
  const toolArgs = (args ?? {}) as Record<string, unknown>;

  try {
    switch (name) {
      case "panel_chat":
        return await handleChat(toolArgs);
      case "panel_code_review":
        return await handleCodeReview(toolArgs);
      case "panel_session_start":
        return await handleSessionStart(toolArgs);
      case "panel_session_message":
        return await handleSessionMessage(toolArgs);
      case "panel_session_end":
        return await handleSessionEnd(toolArgs);
      case "panel_sessions_list":
        return await handleSessionsList();
      case "panel_consult":
        return await handleConsult(toolArgs);
      case "panel_providers":
        return await handleProviders();
      default:
        return errorResult(`Unknown tool: ${name}`);
    }
  } catch (err: unknown) {
    return errorResult(`LLM Panel error: ${extractError(err)}`);
  }
});

// --- Start ---

await mcp.connect(new StdioServerTransport());
