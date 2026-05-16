/**
 * Sandboxed filesystem helpers for agent-node SDK adapters (read-only, cwd-bound).
 */
import fs from "node:fs/promises";
import path from "node:path";

const DEFAULT_MAX_BYTES = 120_000;

export async function safeResolveUnderRoot(
  rootRaw: string,
  relRaw: string
): Promise<{ abs: string } | { error: string }> {
  let root: string;
  try {
    root = await fs.realpath(path.resolve(rootRaw));
  } catch {
    return { error: `workspace root does not exist: ${rootRaw}` };
  }
  const trimmed = relRaw.trim() || ".";
  const normalized = path.normalize(trimmed);
  if (normalized.includes("..")) {
    return { error: "path must not contain .." };
  }
  const joined = path.resolve(root, normalized);
  let abs: string;
  try {
    abs = await fs.realpath(joined);
  } catch {
    return { error: `not found: ${relRaw}` };
  }
  const relative = path.relative(root, abs);
  if (relative.startsWith("..") || path.isAbsolute(relative)) {
    return { error: "path escapes workspace root" };
  }
  return { abs };
}

export async function workspaceReadFile(
  root: string,
  relPath: string,
  maxBytes: number = DEFAULT_MAX_BYTES
): Promise<Record<string, unknown>> {
  const r = await safeResolveUnderRoot(root, relPath);
  if ("error" in r) {
    return { error: r.error };
  }
  const stat = await fs.stat(r.abs);
  if (!stat.isFile()) {
    return { error: `not a file: ${relPath}` };
  }
  if (stat.size > maxBytes) {
    const fh = await fs.open(r.abs, "r");
    try {
      const buf = Buffer.alloc(maxBytes);
      const { bytesRead } = await fh.read(buf, 0, maxBytes, 0);
      const text = buf.subarray(0, bytesRead).toString("utf8");
      return {
        content:
          text +
          `\n\n[truncated: file is ${stat.size} bytes, showing first ${maxBytes} bytes]\n`,
      };
    } finally {
      await fh.close();
    }
  }
  const content = await fs.readFile(r.abs, "utf8");
  return { content };
}

export async function workspaceListDir(
  root: string,
  relPath: string
): Promise<Record<string, unknown>> {
  const rel = relPath.trim() || ".";
  const r = await safeResolveUnderRoot(root, rel);
  if ("error" in r) {
    return { error: r.error };
  }
  const stat = await fs.stat(r.abs);
  if (!stat.isDirectory()) {
    return { error: `not a directory: ${relPath}` };
  }
  const names = await fs.readdir(r.abs);
  const lines: string[] = [];
  let n = 0;
  for (const name of names) {
    if (name.startsWith(".")) continue;
    if (n >= 200) break;
    const p = path.join(r.abs, name);
    try {
      const st = await fs.stat(p);
      lines.push(st.isDirectory() ? `${name}/` : name);
      n += 1;
    } catch {
      lines.push(name);
      n += 1;
    }
  }
  let text = lines.join("\n");
  if (names.length > 200) {
    text += `\n... (${names.length} total entries, listing truncated)`;
  }
  return { entries: text };
}

export { DEFAULT_MAX_BYTES };
