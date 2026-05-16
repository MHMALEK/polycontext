/**
 * Gemini (`@google/genai`) — optional workspace access via CallableTool + AFC
 * (see Google Gen AI JS docs: automatic function calling with CallableTool).
 * @see https://googleapis.github.io/js-genai/
 */
import {
  GoogleGenAI,
  Type,
  FunctionCallingConfigMode,
  createPartFromFunctionResponse,
  type CallableTool,
  type Content,
  type FunctionCall,
  type Part,
  type Tool,
} from "@google/genai";
import {
  workspaceReadFile,
  workspaceListDir,
  DEFAULT_MAX_BYTES,
} from "./workspace_tools.js";

export type GeminiRunBody = {
  systemPrompt?: string;
  prompt: string;
  apiKey: string;
  modelId?: string;
  timeoutSec?: number;
  /** Workspace root on the agent-node host (e.g. REPOS_ROOT); enables CallableTool + AFC. */
  cwd?: string;
  maxToolRounds?: number;
};

function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  return new Promise((resolve, reject) => {
    const t = setTimeout(() => reject(new Error(`timeout after ${ms}ms`)), ms);
    promise.then(
      (v) => {
        clearTimeout(t);
        resolve(v);
      },
      (e) => {
        clearTimeout(t);
        reject(e);
      }
    );
  });
}

const FILE_TOOLS: Tool = {
  functionDeclarations: [
    {
      name: "read_file",
      description:
        "Read a UTF-8 text file under the workspace. Path is relative to the workspace root (e.g. traceability/src/foo.ts).",
      parameters: {
        type: Type.OBJECT,
        properties: {
          path: {
            type: Type.STRING,
            description: "File path relative to workspace root",
          },
          maxBytes: {
            type: Type.INTEGER,
            description: `Optional max bytes (default ${DEFAULT_MAX_BYTES})`,
          },
        },
        required: ["path"],
      },
    },
    {
      name: "list_directory",
      description:
        "List non-hidden files and folders in a directory under the workspace. Path is relative to the workspace root; use . for the root.",
      parameters: {
        type: Type.OBJECT,
        properties: {
          path: {
            type: Type.STRING,
            description: "Directory path relative to workspace root",
          },
        },
        required: ["path"],
      },
    },
  ],
};

async function executeToolCall(
  cwd: string,
  fc: FunctionCall
): Promise<Record<string, unknown>> {
  const name = fc.name ?? "";
  const args = fc.args ?? {};
  if (name === "read_file") {
    const p = String(args.path ?? "");
    const maxB = Number(args.maxBytes);
    const cap =
      Number.isFinite(maxB) && maxB > 0 ? Math.min(maxB, 500_000) : DEFAULT_MAX_BYTES;
    return workspaceReadFile(cwd, p, cap);
  }
  if (name === "list_directory") {
    return workspaceListDir(cwd, String(args.path ?? "."));
  }
  return { error: `unknown tool: ${name}` };
}

/** Implements {@link CallableTool} so the SDK runs the AFC loop (see js-genai docs). */
class WorkspaceFsCallableTool implements CallableTool {
  constructor(private readonly cwd: string) {}

  async tool(): Promise<Tool> {
    return FILE_TOOLS;
  }

  async callTool(functionCalls: FunctionCall[]): Promise<Part[]> {
    const parts: Part[] = [];
    let idx = 0;
    for (const fc of functionCalls) {
      const result = await executeToolCall(this.cwd, fc);
      const id = fc.id?.trim() || `call_${idx++}`;
      parts.push(
        createPartFromFunctionResponse(id, fc.name ?? "unknown", result)
      );
    }
    return parts;
  }
}

function countFunctionResponsesInHistory(history: Content[] | undefined): number {
  if (!history?.length) return 0;
  let n = 0;
  for (const c of history) {
    for (const p of c.parts ?? []) {
      if (p && typeof p === "object" && "functionResponse" in p && p.functionResponse) {
        n += 1;
      }
    }
  }
  return n;
}

export async function runGemini(body: GeminiRunBody): Promise<{
  ok: boolean;
  answer?: string;
  error?: string;
  model?: string;
  durationMs?: number;
  tokensIn?: number;
  tokensOut?: number;
  toolCalls?: number;
}> {
  const started = Date.now();
  const apiKey = body.apiKey;
  if (!apiKey.trim()) {
    return { ok: false, error: "apiKey required" };
  }
  const modelId =
    body.modelId || process.env.GEMINI_SDK_MODEL || "gemini-2.5-flash";
  const timeoutMs = (body.timeoutSec ?? 600) * 1000;
  const cwd = (body.cwd || "").trim();

  if (!cwd) {
    return runGeminiPlain({
      apiKey,
      modelId,
      timeoutMs,
      started,
      prompt: body.prompt,
      systemPrompt: body.systemPrompt,
    });
  }

  const maxRemoteCalls = Math.min(32, Math.max(1, body.maxToolRounds ?? 24));
  const ai = new GoogleGenAI({ apiKey });
  const callable = new WorkspaceFsCallableTool(cwd);

  const promptText = `${body.prompt}\n\n[Workspace root on server: ${cwd}. Use read_file and list_directory to inspect code.]`;

  try {
    const response = await withTimeout(
      ai.models.generateContent({
        model: modelId,
        contents: promptText,
        config: {
          systemInstruction: body.systemPrompt?.trim(),
          tools: [callable],
          toolConfig: {
            functionCallingConfig: { mode: FunctionCallingConfigMode.AUTO },
          },
          automaticFunctionCalling: {
            disable: false,
            maximumRemoteCalls: maxRemoteCalls,
          },
        },
      }),
      timeoutMs
    );

    const um = response.usageMetadata;
    const toolCalls = countFunctionResponsesInHistory(
      response.automaticFunctionCallingHistory
    );

    return {
      ok: true,
      answer: response.text ?? "",
      model: modelId,
      durationMs: Date.now() - started,
      tokensIn: um?.promptTokenCount,
      tokensOut: um?.candidatesTokenCount,
      toolCalls,
    };
  } catch (err) {
    const e = err as Error;
    return {
      ok: false,
      error: `${e?.name || "Error"}: ${e?.message || String(e)}`,
      model: modelId,
      durationMs: Date.now() - started,
    };
  }
}

async function runGeminiPlain(args: {
  apiKey: string;
  prompt: string;
  systemPrompt?: string;
  modelId: string;
  timeoutMs: number;
  started: number;
}): Promise<{
  ok: boolean;
  answer?: string;
  error?: string;
  model?: string;
  durationMs?: number;
  tokensIn?: number;
  tokensOut?: number;
  toolCalls?: number;
}> {
  const ai = new GoogleGenAI({ apiKey: args.apiKey });
  try {
    const response = await withTimeout(
      ai.models.generateContent({
        model: args.modelId,
        contents: args.prompt,
        config: {
          ...(args.systemPrompt?.trim()
            ? { systemInstruction: args.systemPrompt }
            : {}),
        },
      }),
      args.timeoutMs
    );
    const usage = response.usageMetadata;
    return {
      ok: true,
      answer: response.text ?? "",
      model: args.modelId,
      durationMs: Date.now() - args.started,
      tokensIn: usage?.promptTokenCount,
      tokensOut: usage?.candidatesTokenCount,
      toolCalls: 0,
    };
  } catch (err) {
    const e = err as Error;
    return {
      ok: false,
      error: `${e?.name || "Error"}: ${e?.message || String(e)}`,
      model: args.modelId,
      durationMs: Date.now() - args.started,
    };
  }
}
