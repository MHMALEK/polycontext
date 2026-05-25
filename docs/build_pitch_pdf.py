"""Render docs/POLYCONTEXT_PITCH.md to a self-contained HTML page that
prints cleanly to PDF from any browser (Cmd+P → Save as PDF on macOS).

We deliberately don't pull a PDF renderer dependency (weasyprint, etc.)
because the "open in browser, print to PDF" workflow gives the user a
canonical PDF with proper hyphenation/page breaks and avoids a 100 MB
toolchain just for one document.

Run::

    python docs/build_pitch_pdf.py

Output::

    docs/POLYCONTEXT_PITCH.html
"""
from __future__ import annotations

from pathlib import Path

import markdown


HERE = Path(__file__).parent
SRC = HERE / "POLYCONTEXT_PITCH.md"
OUT = HERE / "POLYCONTEXT_PITCH.html"


# CSS tuned for both screen reading and print. Letter/A4 friendly margins,
# consistent typography, code blocks legible at 11pt, tables with subtle
# row dividers. Dark accents (the brand) come through on highlights but
# the page is paper-white so it prints cleanly.
_CSS = """
@page {
  size: A4;
  margin: 22mm 18mm;
}

:root {
  --fg: #0f172a;
  --fg-muted: #475569;
  --fg-subtle: #94a3b8;
  --accent: #16a34a;
  --accent-soft: rgba(22, 163, 74, 0.08);
  --rule: rgba(15, 23, 42, 0.10);
  --warn: #d97706;
  --danger: #dc2626;
  --code-bg: #f4f4f5;
  --table-zebra: #f8fafc;
}

* { box-sizing: border-box; }

html, body {
  margin: 0;
  padding: 0;
  color: var(--fg);
  background: white;
  font-family: 'IBM Plex Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  font-size: 11pt;
  line-height: 1.55;
  -webkit-font-smoothing: antialiased;
}

main {
  max-width: 800px;
  margin: 0 auto;
  padding: 32px 36px 80px;
}

h1, h2, h3, h4 {
  font-family: 'IBM Plex Sans', 'Inter', sans-serif;
  font-weight: 600;
  color: var(--fg);
  letter-spacing: -0.012em;
  line-height: 1.25;
  margin-top: 1.6em;
  margin-bottom: 0.5em;
}

h1 {
  font-size: 32pt;
  margin-top: 0;
  letter-spacing: -0.02em;
  line-height: 1.1;
}

h1 + p { /* subhead */
  font-size: 14pt;
  color: var(--fg-muted);
  margin-top: -8px;
  font-weight: 400;
}

h2 {
  font-size: 18pt;
  border-top: 1px solid var(--rule);
  padding-top: 1.4em;
  margin-top: 2.4em;
}

h3 { font-size: 13pt; margin-top: 1.8em; }
h4 { font-size: 11.5pt; color: var(--fg-muted); text-transform: uppercase; letter-spacing: 0.04em; font-weight: 500; }

p, ul, ol { margin: 0.7em 0; }
ul, ol { padding-left: 1.4em; }
li { margin: 0.25em 0; }
li::marker { color: var(--accent); }

a {
  color: var(--accent);
  text-decoration: none;
  border-bottom: 1px dotted var(--accent);
}

strong { color: var(--fg); font-weight: 600; }
em { color: var(--fg-muted); }

hr {
  border: none;
  border-top: 1px solid var(--rule);
  margin: 2.4em 0;
}

/* Blockquotes — used for the headline + tagline samples in the doc. */
blockquote {
  border-left: 3px solid var(--accent);
  background: var(--accent-soft);
  margin: 1.2em 0;
  padding: 0.6em 1em;
  color: var(--fg);
  border-radius: 0 6px 6px 0;
}
blockquote em {
  color: var(--accent);
  font-style: normal;
  font-weight: 500;
}

/* Code — both inline and blocks. Use IBM Plex Mono if available. */
code, pre, kbd, samp {
  font-family: 'IBM Plex Mono', 'JetBrains Mono', Menlo, Consolas, monospace;
  font-size: 0.92em;
}
code {
  background: var(--code-bg);
  padding: 0.1em 0.4em;
  border-radius: 4px;
  color: var(--fg);
}
pre {
  background: var(--code-bg);
  border: 1px solid var(--rule);
  border-radius: 8px;
  padding: 14px 16px;
  overflow-x: auto;
  line-height: 1.45;
  font-size: 9.5pt;
}
pre code { background: transparent; padding: 0; border-radius: 0; }

/* Tables — borrowed from the in-app aesthetic, light + sparse. */
table {
  border-collapse: collapse;
  width: 100%;
  margin: 1em 0;
  font-size: 10pt;
}
th, td {
  padding: 7px 10px;
  text-align: left;
  border-bottom: 1px solid var(--rule);
  vertical-align: top;
}
th {
  background: var(--table-zebra);
  font-weight: 600;
  color: var(--fg);
  font-size: 9.5pt;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
tbody tr:nth-child(odd) { background: var(--table-zebra); }

/* The TL;DR / WHY NOW heading flair. */
h2:first-of-type {
  border-top: none;
  padding-top: 0;
  margin-top: 1.5em;
}

/* Cover-page-ish first block. */
main > h1 {
  padding-bottom: 0;
  margin-bottom: 6px;
}

/* Print: keep h2 sections together where possible; avoid orphans. */
@media print {
  body { font-size: 10.5pt; }
  h1 { font-size: 28pt; }
  h2 { font-size: 16pt; page-break-after: avoid; }
  h3, h4 { page-break-after: avoid; }
  pre, table, blockquote { page-break-inside: avoid; }
  p, li { orphans: 3; widows: 3; }
}
"""


def build() -> Path:
    md_text = SRC.read_text(encoding="utf-8")
    html_body = markdown.markdown(
        md_text,
        extensions=[
            "extra",       # tables, fenced code, attr_list, def_lists
            "smarty",      # smart quotes
            "sane_lists",  # consistent list parsing
        ],
        output_format="html5",
    )
    html_doc = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>Polycontext — pitch</title>
  <link href=\"https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap\" rel=\"stylesheet\">
  <style>{_CSS}</style>
</head>
<body>
<main>
{html_body}
</main>
</body>
</html>
"""
    OUT.write_text(html_doc, encoding="utf-8")
    return OUT


if __name__ == "__main__":
    out = build()
    print(f"wrote {out}")
    print(f"open with: open {out}")
    print("then Cmd+P → 'Save as PDF' for a print-ready document.")
