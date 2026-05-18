/**
 * Env-driven config. Reads .env via Node's built-in `--env-file` flag (run with
 * `tsx --env-file=.env src/cli.ts`) or rely on the host shell to export them.
 *
 * Kept tiny on purpose — no schema lib, no validators. The bridge is meant to
 * be lifted into its own repo and stay easy to read.
 */

function required(name: string): string {
  const v = process.env[name]?.trim();
  if (!v) {
    throw new Error(`${name} is required (see .env.example)`);
  }
  return v;
}

function optional(name: string, fallback = ""): string {
  return process.env[name]?.trim() || fallback;
}

export type Config = {
  jira: {
    baseUrl: string;
    email: string;
    apiToken: string;
  };
  techDecomp: {
    url: string;
    adapter: string;
    repos: string[] | undefined;
    timeoutSeconds: number;
  };
  server: {
    port: number;
    host: string;
    sharedSecret: string;
  };
};

export function loadConfig(): Config {
  return {
    jira: {
      baseUrl: required("JIRA_BASE_URL").replace(/\/+$/, ""),
      email: required("JIRA_EMAIL"),
      apiToken: required("JIRA_API_TOKEN"),
    },
    techDecomp: {
      url: optional("TECH_DECOMP_URL", "http://localhost:18000").replace(/\/+$/, ""),
      adapter: optional("TECH_DECOMP_ADAPTER", "cursor"),
      repos: optional("TECH_DECOMP_REPOS")
        ? optional("TECH_DECOMP_REPOS")
            .split(",")
            .map((s) => s.trim())
            .filter(Boolean)
        : undefined,
      timeoutSeconds: Number(optional("TECH_DECOMP_TIMEOUT_SECONDS", "600")),
    },
    server: {
      port: Number(optional("PORT", "13200")),
      host: optional("HOST", "0.0.0.0"),
      sharedSecret: optional("JIRA_BRIDGE_SHARED_SECRET"),
    },
  };
}
