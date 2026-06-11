# LLM Panel

A small MCP server that lets Claude (or any MCP client) ask other models what they think.

The idea is simple. Claude is great, but sometimes you want a second opinion from a model that thinks differently. So instead of copy-pasting your code into three different chat windows, you ask the panel. One tool call goes out to Kimi, DeepSeek, whatever you've wired up, and you get their answers back.

I built this because I kept catching real bugs by bouncing the same question off different models. They disagree in useful ways. Where they all agree you're probably fine. Where they split, that's usually where the interesting problem is hiding.

Ships with Kimi (Moonshot) and DeepSeek out of the box. MiMo is stubbed in and ready, you just point it at an endpoint. Anything else that speaks the OpenAI API format is basically one block of config away (there's a section on that below).

## What you need

- [Bun](https://bun.sh) — the server runs on Bun, not Node
- An MCP client. Claude Code is what I use it with.
- An API key for at least one provider. You don't need all of them. No key just means that provider gets skipped and the server still starts fine.

## Install

Clone it and grab the deps:

```bash
git clone https://github.com/dominiclynchwoodlands-ui/LLM-Panel.git
cd LLM-Panel
bun install
```

Now the keys, and this is the part to actually read. Your API keys go in a `.env` file that sits next to the code and never gets committed. They are never written into the source anywhere. Copy the example and fill in whatever you've got:

```bash
cp .env.example .env
chmod 600 .env   # lock it down so only your user can read it
```

Open `.env` and paste your keys in:

```
KIMI_API_KEY=sk-your-kimi-key
DEEPSEEK_API_KEY=sk-your-deepseek-key
```

That's the keys done. `.env` is already in `.gitignore` so you can't push it by accident, but do yourself a favor and double check before your first commit anyway.

## Wire it into Claude Code

Point Claude Code at the launcher:

```bash
claude mcp add llm-panel /absolute/path/to/LLM-Panel/launch.sh
```

Or if you'd rather edit the config by hand, drop this into your `mcpServers` block:

```json
"llm-panel": {
  "command": "/absolute/path/to/LLM-Panel/launch.sh",
  "args": [],
  "env": {
    "LLM_PANEL_ENV_FILE": "/absolute/path/to/LLM-Panel/.env"
  }
}
```

`launch.sh` loads your `.env` and then starts the server. It looks for `.env` right next to itself by default, so if you keep the folder layout as-is you can drop the `LLM_PANEL_ENV_FILE` line entirely. Restart your client and the `panel_*` tools show up.

Quick sanity check once it's loaded: ask it to run `panel_providers`. You'll see which models came up available and which got skipped.

## The tools

- `panel_chat` — ask one provider a one-off question
- `panel_code_review` — hand one provider some code plus what to focus on, get a structured review back
- `panel_consult` — the fun one. Same prompt fired at several providers at once, answers come back together. One model timing out or erroring doesn't sink the rest, you just get an error in that slot and everyone else's answers still land.
- `panel_session_start` / `panel_session_message` / `panel_session_end` — a real back-and-forth with one provider that remembers the conversation
- `panel_sessions_list` — what sessions are currently open
- `panel_providers` — who's wired up and available right now

Every tool that talks to a model takes a `provider` argument (`kimi`, `deepseek`, and so on). `panel_consult` takes a list of them, or you leave it off and it just asks everyone that's available.

## Adding another model

If the model speaks the OpenAI API format, and most do now, adding it takes a minute. Open `providers.ts`, find the registry, and add an entry alongside the others:

```ts
{
  id: "yourmodel",
  label: "Your Model",
  baseURL: "https://api.yourmodel.com/v1",
  apiKeyEnv: ["YOURMODEL_API_KEY"],
  defaultModel: "whatever-the-model-id-is",
  // plus the per-provider limits — copy the shape from the kimi/deepseek entries
}
```

Then put `YOURMODEL_API_KEY=...` in your `.env`, restart, and it shows up in `panel_providers`.

One thing worth knowing: temperature is handled per provider on purpose. Kimi flat out rejects anything except temperature=1, so each provider carries its own rule instead of one global setting fighting all of them. If your new model is picky the same way, that's where you'd set it.

## Config / env vars

Keys are the only thing you actually have to set. Everything else has a default. If you want to tune a provider, swap the `KIMI` prefix below for `DEEPSEEK`, `MIMO`, etc:

| var | what it does | default |
|-----|--------------|---------|
| `KIMI_API_KEY` | the key (Kimi also takes `MOONSHOT_API_KEY`) | none |
| `KIMI_MODEL` | which model id to use | `kimi-k2.6` |
| `KIMI_BASE_URL` | override the endpoint | provider default |
| `KIMI_TIMEOUT_MS` | request timeout, kept long on purpose so big prompts don't get cut off | `1800000` (30 min) |
| `KIMI_MAX_OUTPUT_TOKENS` | cap on response length | provider max |
| `KIMI_CONTEXT_TOKENS` | context window hint | provider default |

DeepSeek defaults to `deepseek-v4-pro` with a 1M token context window. MiMo needs `MIMO_BASE_URL` set since there's no public endpoint baked in for it.

The long default timeout is deliberate. Streaming stays on for every call, which keeps the socket warm, so large requests don't drop halfway through. I got bitten by that early on and never want to debug it again.

## About the keys, since people always ask

No key ever lives in the code. The server reads them from the environment at startup, and `launch.sh` is the only thing that loads your `.env`. Keep that file at `chmod 600`, keep it out of git (it already is), and you're set. If you point `LLM_PANEL_ENV_FILE` at some shared env file that has other secrets in it, just know the server process will see those too, so a dedicated `.env` is the cleaner move.

## License

MIT. Do whatever you want with it. Fork it, tear it apart, build your own thing on top. That's the whole point.
