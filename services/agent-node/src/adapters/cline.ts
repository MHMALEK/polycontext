/** Cline SDK — runs inside services/agent-node. */

import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";

import { Agent, createTool } from "@cline/sdk";

export type ClineRunBody = {
  systemPrompt?: string;
  prompt: string;
  cwd?: string;
  providerId: string;
  modelId: string;
  apiKey: string;
  maxIterations?: number;
  timeoutSec?: number;
  enableFindCode?: boolean;
};

const SKIP_DIR_NAMES = new Set([
  ".git",
  "node_modules",
  ".venv",
  "venv",
  "__pycache__",
  ".tox",
  "dist",
  "build",
  ".next",
  "target",
  ".cargo",
]);

function resolveUnderRoot(root: string, rel: string): string {
  const resolved = path.resolve(root, path.normalize(rel));
  const prefix = root.endsWith(path.sep) ? root : root + path.sep;
  if (resolved !== root && !resolved.startsWith(prefix)) {
    throw new Error("path escapes workspace root");
  }
  return resolved;
}

function createWorkspaceTools(workspaceRoot: string) {
  const readWorkspaceFileTool = createTool({
    name: "read_workspace_file",
    description:
      "Read a UTF-8 text file under the task working directory. Use this (or grep_workspace) " +
      "to ground answers in real code. Path is relative to the workspace root.",
    inputSchema: {
      type: "object",
      properties: {
        file_path: {
          type: "string",
          description: "Path relative to workspace root (e.g. src/app/page.tsx).",
        },
        max_bytes: {
          type: "integer",
          description: "Max bytes to return (default 200000).",
        },
      },
      required: ["file_path"],
    },
    execute: async ({
      file_path,
      max_bytes,
    }: {
      file_path: string;
      max_bytes?: number;
    }) => {
      const cap = Math.min(Math.max(1024, max_bytes ?? 200_000), 500_000);
      try {
        const abs = resolveUnderRoot(workspaceRoot, file_path);
        const st = await fsp.stat(abs);
        if (!st.isFile()) return { text: `Not a file: ${file_path}` };
        if (st.size > cap) {
          const buf = Buffer.alloc(cap);
          const fh = await fsp.open(abs, "r");
          try {
            await fh.read(buf, 0, cap, 0);
          } finally {
            await fh.close();
          }
          const slice = buf.toString("utf8");
          return {
            text:
              `(truncated first ${cap} bytes of ${file_path}, file size ${st.size})\n` + slice,
          };
        }
        const text = await fsp.readFile(abs, "utf8");
        return { text };
      } catch (e) {
        const err = e as Error;
        return { text: `read_workspace_file error: ${err?.message || String(e)}` };
      }
    },
  });

  const grepWorkspaceTool = createTool({
    name: "grep_workspace",
    description:
      "Search files under the workspace for a literal string or regex. Returns matching lines " +
      "with file paths and line numbers. Use before claiming how code works.",
    inputSchema: {
      type: "object",
      properties: {
        pattern: { type: "string", description: "Substring or regex pattern to search for." },
        is_regex: {
          type: "boolean",
          description: "If true, pattern is a JavaScript regex (use carefully). Default false.",
        },
        path_prefix: {
          type: "string",
          description: "Optional subdirectory relative to workspace to narrow search.",
        },
        max_matches: {
          type: "integer",
          description: "Stop after this many matches (default 40, max 80).",
        },
      },
      required: ["pattern"],
    },
    execute: async ({
      pattern,
      is_regex,
      path_prefix,
      max_matches,
    }: {
      pattern: string;
      is_regex?: boolean;
      path_prefix?: string;
      max_matches?: number;
    }) => {
      const cap = Math.min(Math.max(1, max_matches ?? 40), 80);
      if (!pattern.trim()) return { text: "grep_workspace: empty pattern." };
      let re: RegExp | null = null;
      if (is_regex) {
        try {
          re = new RegExp(pattern, "m");
        } catch (e) {
          return { text: `grep_workspace: invalid regex: ${(e as Error).message}` };
        }
      }

      const searchRoot = path_prefix?.trim()
        ? resolveUnderRoot(workspaceRoot, path_prefix)
        : workspaceRoot;

      const linesOut: string[] = [];
      let matches = 0;
      let filesScanned = 0;
      const maxFiles = 1200;
      const maxFileBytes = 400_000;

      async function scanFile(absFile: string): Promise<void> {
        if (matches >= cap) return;
        let raw: Buffer;
        try {
          raw = await fsp.readFile(absFile);
        } catch {
          return;
        }
        if (raw.includes(0)) return;
        if (raw.length > maxFileBytes) return;
        const text = raw.toString("utf8");
        const rel = path.relative(workspaceRoot, absFile) || absFile;
        const textLines = text.split(/\r?\n/);
        for (let i = 0; i < textLines.length; i++) {
          if (matches >= cap) break;
          const line = textLines[i];
          const hit = re ? re.test(line) : line.includes(pattern);
          if (re) re.lastIndex = 0;
          if (hit) {
            matches += 1;
            linesOut.push(`${rel}:${i + 1}:${line}`);
          }
        }
      }

      async function walk(dir: string): Promise<void> {
        if (matches >= cap || filesScanned >= maxFiles) return;
        let entries: fs.Dirent[];
        try {
          entries = await fsp.readdir(dir, { withFileTypes: true });
        } catch {
          return;
        }
        for (const ent of entries) {
          if (matches >= cap || filesScanned >= maxFiles) break;
          const abs = path.join(dir, ent.name);
          if (ent.isDirectory()) {
            if (SKIP_DIR_NAMES.has(ent.name)) continue;
            await walk(abs);
          } else if (ent.isFile()) {
            filesScanned += 1;
            await scanFile(abs);
          }
        }
      }

      try {
        await walk(searchRoot);
      } catch (e) {
        return { text: `grep_workspace error: ${(e as Error).message || String(e)}` };
      }
      if (matches === 0) {
        return {
          text: `grep_workspace: no matches for ${JSON.stringify(pattern)} under ${path.relative(workspaceRoot, searchRoot) || "."} (scanned ${filesScanned} files).`,
        };
      }
      return { text: linesOut.join("\n") };
    },
  });

  return [readWorkspaceFileTool, grepWorkspaceTool];
}

const findCodeTool = createTool({
  name: "find_code",
  description:
    "Search the codebase for relevant files. Use this BEFORE making claims about how the code works. " +
    "Returns up to N snippets with file paths, line ranges, and code excerpts.",
  inputSchema: {
    type: "object",
    properties: {
      query: { type: "string", description: "Search query — keywords, symbols, or short phrases." },
      max_results: {
        type: "integer",
        description: "Cap on results returned. Default 10, max 30.",
      },
    },
    required: ["query"],
  },
  execute: async ({ query, max_results }: { query: string; max_results?: number }) => {
    const url = (process.env.SOURCEBOT_URL || "").replace(/\/+$/, "");
    const key = process.env.SOURCEBOT_API_KEY || "";
    if (!url) return { text: "find_code unavailable: SOURCEBOT_URL not set." };
    const cap = Math.min(Math.max(1, max_results ?? 10), 30);
    try {
      const r = await fetch(`${url}/api/search`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(key ? { Authorization: `Bearer ${key}` } : {}),
        },
        body: JSON.stringify({
          query,
          matches: cap,
          contextLines: 6,
          isRegexEnabled: false,
          isCaseSensitivityEnabled: false,
        }),
      });
      if (!r.ok)
        return {
          text: `find_code: sourcebot returned ${r.status}: ${(await r.text()).slice(0, 300)}`,
        };
      const data = (await r.json()) as {
        files?: Array<{
          repository?: string;
          fileName?: { text?: string } | string;
          chunkMatches?: Array<Record<string, unknown>>;
          chunks?: Array<Record<string, unknown>>;
          matches?: Array<Record<string, unknown>>;
        }>;
      };
      const lines: string[] = [];
      let n = 0;
      for (const f of data.files || []) {
        const chunks = f.chunkMatches || f.chunks || f.matches || [];
        for (const m of chunks) {
          n += 1;
          const mm = m as Record<string, unknown>;
          const start =
            (mm.rangeStart as { lineNumber?: number } | undefined)?.lineNumber ??
            mm.lineNumber ??
            "?";
          const end =
            (mm.rangeEnd as { lineNumber?: number } | undefined)?.lineNumber ?? start;
          const content = String(mm.content || mm.text || "").trimEnd();
          const fn =
            typeof f.fileName === "object" && f.fileName?.text
              ? f.fileName.text
              : String(f.fileName || "");
          lines.push(`[${n}] ${f.repository || "?"} :: ${fn}:${start}-${end}`);
          lines.push(content);
          lines.push("");
          if (n >= cap) break;
        }
        if (n >= cap) break;
      }
      if (n === 0) return { text: `find_code: no matches for ${JSON.stringify(query)}.` };
      return { text: lines.join("\n") };
    } catch (e) {
      const err = e as Error;
      return { text: `find_code error: ${err?.message || String(e)}` };
    }
  },
});

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

export async function runCline(body: ClineRunBody): Promise<{
  ok: boolean;
  answer?: string;
  error?: string;
  tokensIn?: number;
  tokensOut?: number;
  cacheReadTokens?: number;
  cacheWriteTokens?: number;
  costUsd?: number;
  events?: unknown[];
  status?: string;
  iterations?: number;
}> {
  const events: unknown[] = [];
  const originalCwd = process.cwd();
  if (body.cwd) {
    try {
      process.chdir(body.cwd);
    } catch {
      /* let SDK fail */
    }
  }

  try {
    const tools = [];
    const wsRoot = body.cwd?.trim() ? path.resolve(body.cwd.trim()) : "";
    if (wsRoot) {
      try {
        const st = await fsp.stat(wsRoot);
        if (st.isDirectory()) {
          tools.push(...createWorkspaceTools(wsRoot));
        }
      } catch {
        /* invalid cwd — omit workspace tools */
      }
    }
    if (body.enableFindCode && (process.env.SOURCEBOT_URL || "").trim()) {
      tools.push(findCodeTool);
    }

    if (tools.length === 0) {
      const sb = (process.env.SOURCEBOT_URL || "").trim();
      const parts = [
        wsRoot
          ? `Workspace path is not a readable directory on this agent-node process: ${wsRoot}`
          : "No workspace cwd was sent.",
      ];
      if (body.enableFindCode && !sb) {
        parts.push("SOURCEBOT_URL is unset, so find_code was not registered.");
      }
      parts.push(
        "Run agent-node where Python's cwd exists (same machine), or mount REPOS_ROOT into the agent-node container at the same absolute path."
      );
      return {
        ok: false,
        error: parts.join(" "),
        events: [],
      };
    }

    const agent = new Agent({
      providerId: body.providerId,
      modelId: body.modelId,
      apiKey: body.apiKey,
      maxIterations: body.maxIterations ?? 30,
      systemPrompt: body.systemPrompt,
      tools,
      // Headless agent-node runs must execute tools without an IDE approval UI.
      toolPolicies: {
        "*": { enabled: true, autoApprove: true },
      },
      requestToolApproval: async () => ({ approved: true }),
    });

    if (typeof agent.subscribe === "function") {
      agent.subscribe((event: unknown) => {
        events.push(event);
        if (events.length > 200) events.shift();
      });
    }

    const result = (await withTimeout(
      agent.run(body.prompt),
      (body.timeoutSec ?? 600) * 1000
    )) as {
      outputText?: string;
      status?: string;
      iterations?: number;
      usage?: {
        inputTokens?: number;
        outputTokens?: number;
        cacheReadTokens?: number;
        cacheWriteTokens?: number;
        totalCost?: number;
      };
      error?: { message?: string } | string;
    };

    const usage = result?.usage ?? {};
    const finalText = result?.outputText ?? "";

    return {
      ok: true,
      answer: finalText,
      status: result?.status,
      iterations: result?.iterations,
      tokensIn: usage.inputTokens,
      tokensOut: usage.outputTokens,
      cacheReadTokens: usage.cacheReadTokens,
      cacheWriteTokens: usage.cacheWriteTokens,
      costUsd: usage.totalCost,
      events: events.slice(-20),
      error: result?.error
        ? String(
            typeof result.error === "object" && result.error?.message
              ? result.error.message
              : result.error
          )
        : undefined,
    };
  } catch (err) {
    const e = err as Error;
    return {
      ok: false,
      error: `${e?.name || "Error"}: ${e?.message || String(e)}`,
      events: events.slice(-20),
    };
  } finally {
    if (body.cwd) {
      try {
        process.chdir(originalCwd);
      } catch {
        /* ignore */
      }
    }
  }
}
