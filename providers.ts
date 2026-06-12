#!/usr/bin/env bun

import OpenAI from "openai";

// --- Provider Registry ---

export interface Provider {
  id: string;
  label: string;
  /** Static base URL for this provider. null means it MUST come from <ID>_BASE_URL env. */
  baseURL: string | null;
  /** First env var that is present and non-empty wins as the API key. */
  apiKeyEnv: string[];
  defaultModel: string;
  defaultModelEnv: string;
  /**
   * Temperature to send with every request.
   * kimi / kimi-code: 1 (k2.6 and k2.7-code both require it — any other value → HTTP 400).
   * null: omit temperature entirely (let the API use its default).
   */
  temperature: number | null;

  // --- Capacity limits (all have per-provider env overrides: <ID>_*) ---

  /** Documented context window size in tokens (for reference and guards). */
  contextWindowTokens: number;
  /** Completion token cap sent as max_tokens in every request. */
  maxOutputTokens: number;
  /** Hard model maximum — maxOutputTokens is clamped to this ceiling. */
  maxOutputCeiling: number;
  /** Max accumulated chars across all turns in a session. */
  maxSessionChars: number;
  /** Max chars per single message / input. */
  maxInputChars: number;
  /** Per-request timeout in ms. Keeps the socket warm for long streaming responses. */
  timeoutMs: number;
  /** Max retries on transient failures. */
  maxRetries: number;

  // Env var names (for documentation/error messages)
  maxOutputTokensEnv: string;
}

function envNum(key: string, fallback: number): number {
  const v = process.env[key];
  if (v !== undefined) {
    const n = Number(v);
    if (!isNaN(n)) return n;
  }
  return fallback;
}

function clamp(value: number, max: number): number {
  return Math.min(value, max);
}

// Env-var prefix for a provider id: non-alphanumerics → "_" so an id containing a
// hyphen (e.g. "kimi-code") maps to a VALID env name (KIMI_CODE_*) instead of an
// illegal one (KIMI-CODE_* — which the shell can't export and process.env never sees).
export function envPrefix(id: string): string {
  return id.toUpperCase().replace(/[^A-Z0-9]/g, "_");
}

// Build a provider entry resolving all per-provider env overrides.
// Pattern: <ID>_<FIELD> — e.g. KIMI_MAX_OUTPUT_TOKENS, DEEPSEEK_TIMEOUT_MS
function makeProvider(base: {
  id: string;
  label: string;
  baseURL: string | null;
  apiKeyEnv: string[];
  defaultModel: string;
  defaultModelEnv: string;
  temperature: number | null;
  contextWindowTokens: number;
  maxOutputTokens: number;
  maxOutputCeiling: number;
  maxSessionChars: number;
  maxInputChars: number;
  timeoutMs: number;
  maxRetries: number;
}): Provider {
  const ID = envPrefix(base.id);
  const raw = envNum(`${ID}_MAX_OUTPUT_TOKENS`, base.maxOutputTokens);
  const ceiling = base.maxOutputCeiling;
  return {
    ...base,
    // Use || not ?? so that an empty string env var (e.g. KIMI_MODEL="") falls through
    // to the registry default rather than propagating an empty model id.
    defaultModel: process.env[base.defaultModelEnv] || base.defaultModel,
    maxOutputTokens: Math.max(1, clamp(raw, ceiling)),
    maxOutputCeiling: ceiling,
    contextWindowTokens: envNum(`${ID}_CONTEXT_TOKENS`, base.contextWindowTokens),
    maxSessionChars: envNum(`${ID}_MAX_SESSION_CHARS`, base.maxSessionChars),
    maxInputChars: envNum(`${ID}_MAX_INPUT_CHARS`, base.maxInputChars),
    timeoutMs: envNum(`${ID}_TIMEOUT_MS`, base.timeoutMs),
    maxRetries: envNum(`${ID}_MAX_RETRIES`, base.maxRetries),
    maxOutputTokensEnv: `${ID}_MAX_OUTPUT_TOKENS`,
  };
}

export const PROVIDERS: Provider[] = [
  makeProvider({
    id: "kimi",
    label: "Kimi K2.6 (Moonshot · general)",
    baseURL: "https://api.moonshot.ai/v1",
    apiKeyEnv: ["KIMI_API_KEY", "MOONSHOT_API_KEY"],
    defaultModel: "kimi-k2.6",
    defaultModelEnv: "KIMI_MODEL",
    // k2.6 hard-requires temperature=1 — any other value returns HTTP 400.
    // This is a PER-PROVIDER setting; never globalize it.
    temperature: 1,
    contextWindowTokens: 262_144,
    maxOutputTokens: 131_072,
    maxOutputCeiling: 131_072,
    maxSessionChars: 1_500_000,
    maxInputChars: 500_000,
    timeoutMs: 1_800_000,
    maxRetries: 2,
  }),
  makeProvider({
    id: "kimi-code",
    label: "Kimi K2.7 Code (Moonshot)",
    baseURL: "https://api.moonshot.ai/v1",
    // Shares the same Moonshot account/key as `kimi` — set KIMI_API_KEY (or
    // MOONSHOT_API_KEY) once and both providers light up. Override the model
    // independently with KIMI_CODE_MODEL.
    apiKeyEnv: ["KIMI_CODE_API_KEY", "KIMI_API_KEY", "MOONSHOT_API_KEY"],
    defaultModel: "kimi-k2.7-code",
    defaultModelEnv: "KIMI_CODE_MODEL",
    // k2.7-code is always-thinking and requires temperature=1 (same as k2.6).
    temperature: 1,
    contextWindowTokens: 262_144,
    maxOutputTokens: 131_072,
    maxOutputCeiling: 131_072,
    maxSessionChars: 1_500_000,
    maxInputChars: 500_000,
    timeoutMs: 1_800_000,
    maxRetries: 2,
  }),
  makeProvider({
    id: "deepseek",
    label: "DeepSeek",
    baseURL: "https://api.deepseek.com",
    apiKeyEnv: ["DEEPSEEK_API_KEY"],
    defaultModel: "deepseek-v4-pro",
    defaultModelEnv: "DEEPSEEK_MODEL",
    // null → omit temperature field entirely so the API uses its own default.
    temperature: null,
    contextWindowTokens: 1_048_576,
    maxOutputTokens: 65_536,
    maxOutputCeiling: 384_000,
    maxSessionChars: 3_000_000,
    maxInputChars: 2_000_000,
    timeoutMs: 1_800_000,
    maxRetries: 2,
  }),
  makeProvider({
    id: "mimo",
    label: "MiMo (Xiaomi)",
    // Public Xiaomi MiMo endpoint. Override with MIMO_BASE_URL for a private deployment.
    baseURL: "https://api.xiaomimimo.com/v1",
    apiKeyEnv: ["MIMO_API_KEY"],
    // Flagship ("max") model on the MiMo API. Note: mimo-7b is the open-weights
    // HF checkpoint, NOT a valid API model id — the API serves the mimo-v2* family.
    defaultModel: "mimo-v2.5-pro",
    defaultModelEnv: "MIMO_MODEL",
    temperature: null,
    contextWindowTokens: 262_144,
    maxOutputTokens: 65_536,
    maxOutputCeiling: 65_536,
    maxSessionChars: 1_000_000,
    maxInputChars: 500_000,
    timeoutMs: 1_800_000,
    maxRetries: 2,
  }),
];

// Build a fast lookup map.
const REGISTRY = new Map<string, Provider>(PROVIDERS.map((p) => [p.id, p]));

export function getProvider(id: string): Provider | undefined {
  return REGISTRY.get(id);
}

// --- Resolution helpers ---

export function resolveApiKey(p: Provider): string | null {
  for (const envVar of p.apiKeyEnv) {
    const val = process.env[envVar];
    if (val && !val.startsWith("YOUR_")) return val;
  }
  return null;
}

export function resolveBaseURL(p: Provider): string | null {
  const override = process.env[`${envPrefix(p.id)}_BASE_URL`];
  if (override) return override;
  return p.baseURL;
}

export function isAvailable(p: Provider): boolean {
  return resolveBaseURL(p) !== null && resolveApiKey(p) !== null;
}

/** Human-readable reason why a provider is unavailable (or undefined if available). */
export function unavailableReason(p: Provider): string | undefined {
  const key = resolveApiKey(p);
  const base = resolveBaseURL(p);
  if (!key && !base) return "no API key and no base URL";
  if (!key) return "no API key";
  if (!base) return "no base URL";
  return undefined;
}

export interface ProviderStatus {
  id: string;
  label: string;
  defaultModel: string;
  available: boolean;
  baseURL: string | null;
  reason?: string;
}

export function listProviders(): ProviderStatus[] {
  return PROVIDERS.map((p) => ({
    id: p.id,
    label: p.label,
    defaultModel: p.defaultModel,
    available: isAvailable(p),
    baseURL: resolveBaseURL(p),
    reason: unavailableReason(p),
  }));
}

// --- Client cache ---

const clientCache = new Map<string, OpenAI>();

export function getClient(id: string): OpenAI {
  const p = REGISTRY.get(id);
  if (!p) throw new Error(`Unknown provider: ${id}`);

  const base = resolveBaseURL(p);
  const key = resolveApiKey(p);
  if (!base) throw new Error(`Provider '${id}' (${p.label}): no base URL configured. Set ${envPrefix(id)}_BASE_URL.`);
  if (!key) throw new Error(`Provider '${id}' (${p.label}): no API key configured. Set one of: ${p.apiKeyEnv.join(", ")}.`);

  const cached = clientCache.get(id);
  if (cached) return cached;

  const client = new OpenAI({ apiKey: key, baseURL: base, maxRetries: p.maxRetries });
  clientCache.set(id, client);
  return client;
}

// --- Centralised API call ---

export interface CallOptions {
  model?: string;
}

export interface CallResult {
  content: string;
  reasoning: string;
  usage: OpenAI.Completions.CompletionUsage | undefined;
  finishReason: string | null;
  model: string;
}

/**
 * The SINGLE place that invokes the OpenAI SDK.
 *
 * Per-provider behaviour:
 *   - temperature: sent only when Provider.temperature !== null (kimi always sends 1;
 *     deepseek/mimo omit it so the API uses its own default — avoids HTTP 400 on strict APIs).
 *   - max_tokens: always sent — prevents silent API truncation. Clamped to maxOutputCeiling.
 *   - timeout: per-provider (default 30 min) — keeps socket warm during long reasoning.
 *   - streaming: always enabled to collect reasoning_content then content.
 *   - maxRetries: per-provider, applied to the OpenAI client.
 */
export async function callProvider(
  id: string,
  messages: OpenAI.Chat.ChatCompletionMessageParam[],
  opts: CallOptions = {}
): Promise<CallResult> {
  const p = REGISTRY.get(id);
  if (!p) throw new Error(`Unknown provider: ${id}`);

  const client = getClient(id);
  const model = opts.model ?? p.defaultModel;

  const requestParams: OpenAI.Chat.ChatCompletionCreateParamsStreaming = {
    model,
    messages,
    max_tokens: p.maxOutputTokens,
    stream: true,
    stream_options: { include_usage: true },
    // temperature is ONLY set when the provider requires it (e.g. kimi=1).
    // For providers with temperature:null we omit the field entirely so the
    // API applies its own default — this avoids HTTP 400 on strict APIs and
    // unexpected behaviour on providers that don't expect the field.
    ...(p.temperature !== null ? { temperature: p.temperature } : {}),
  };

  const stream = await client.chat.completions.create(requestParams, {
    timeout: p.timeoutMs,
  });

  let content = "";
  let reasoning = "";
  let usage: OpenAI.Completions.CompletionUsage | undefined;
  let finishReason: string | null = null;

  for await (const chunk of stream) {
    const choice = chunk.choices[0];
    if (choice) {
      content += choice.delta?.content ?? "";
      reasoning +=
        (choice.delta as { reasoning_content?: string })?.reasoning_content ?? "";
      if (choice.finish_reason) finishReason = choice.finish_reason;
    }
    if (chunk.usage) usage = chunk.usage;
  }

  return { content, reasoning, usage, finishReason, model };
}
