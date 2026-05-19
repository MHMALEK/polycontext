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
  workspaceSearchFiles,
  workspaceGrepSearch,
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
    {
      name: "search_files",
      description: "Search for files by name pattern in the workspace.",
      parameters: {
        type: Type.OBJECT,
        properties: {
          pattern: {
            type: Type.STRING,
            description: "The name pattern to search for (e.g. 'user' or '.ts')",
          },
        },
        required: ["pattern"],
      },
    },
    {
      name: "grep_search",
      description: "Search inside file contents for a specific string or regex pattern in the workspace.",
      parameters: {
        type: Type.OBJECT,
        properties: {
          query: {
            type: Type.STRING,
            description: "The string or regex pattern to search for inside files",
          },
        },
        required: ["query"],
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
  if (name === "search_files") {
    return workspaceSearchFiles(cwd, String(args.pattern ?? ""));
  }
  if (name === "grep_search") {
    return workspaceGrepSearch(cwd, String(args.query ?? ""));
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
    body.modelId || process.env.GEMINI_SDK_MODEL || "gemini-2.5-pro";
  const timeoutMs = (body.timeoutSec ?? 600) * 1000;
  const cwd = (body.cwd || "").trim();

  if (!cwd) {
    const ai = new GoogleGenAI({ apiKey });
    try {
      const response = await ai.models.generateContent({
        model: modelId,
        contents: body.prompt,
        config: {
          ...(body.systemPrompt?.trim()
            ? { systemInstruction: body.systemPrompt }
            : {}),
          thinkingConfig: { thinkingBudget: -1, includeThoughts: false },
          temperature: 0.2,
          httpOptions: { timeout: timeoutMs },
        },
      });
      const usage = response.usageMetadata;
      return {
        ok: true,
        answer: response.text ?? "",
        model: modelId,
        durationMs: Date.now() - started,
        tokensIn: usage?.promptTokenCount,
        tokensOut: usage?.candidatesTokenCount,
        toolCalls: 0,
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

  // Bumped from 24 → 48 default to give Gemini room to do thorough multi-step
  // exploration on complex tickets. Cursor's composer-2 routinely makes 25–30
  // tool calls per question in our eval; capping Gemini at 24 was leaving it
  // truncated mid-investigation on the harder cases.
  const maxRemoteCalls = Math.min(80, Math.max(1, body.maxToolRounds ?? 48));
  const ai = new GoogleGenAI({ apiKey });
  const callable = new WorkspaceFsCallableTool(cwd);

  const promptText = `${body.prompt}\n\n[Workspace root on server: ${cwd}. Use read_file, list_directory, search_files, and grep_search to inspect code.]`;

  try {
    const response = await ai.models.generateContent({
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
        // Extended thinking ON with an automatic budget. Gemini 2.5 Pro has
        // thinking but the default budget is conservative — without explicitly
        // setting -1 (automatic, model decides) it under-reasons on
        // multi-step code questions. This matches what gemini-cli and
        // Sourcebot's Gemini integration do internally.
        thinkingConfig: {
          thinkingBudget: -1,
          includeThoughts: false,
        },
        // Lower temperature gives more grounded answers — less inclined to
        // invent file paths. The forced-tool-use system prompt still does
        // the heavy lifting; this just trims the variance.
        temperature: 0.2,
        httpOptions: { timeout: timeoutMs },
      },
    });

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
