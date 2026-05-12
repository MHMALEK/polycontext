"""Follow imports one hop from anchor files.

When we read a file as an anchor, parse its imports and pull in the files
those imports resolve to.

Python: uses stdlib `ast` for accuracy.
TypeScript / JS: tree-sitter when available (handles dynamic imports,
re-exports, multi-line imports robustly), falling back to a regex parser
when tree-sitter isn't importable."""
from __future__ import annotations

import ast
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

# Lazy tree-sitter init. Set on first TS parse.
_TS_PARSER = None
_TS_PARSER_INIT_TRIED = False


def _get_ts_parser():
    global _TS_PARSER, _TS_PARSER_INIT_TRIED
    if _TS_PARSER_INIT_TRIED:
        return _TS_PARSER
    _TS_PARSER_INIT_TRIED = True
    try:
        from tree_sitter import Language, Parser  # type: ignore
        import tree_sitter_typescript as tsts  # type: ignore
        # tree-sitter-typescript ships two grammars: typescript (plain .ts) and tsx.
        # tsx is a superset that also accepts .ts, so we use it for both.
        _TS_PARSER = Parser(Language(tsts.language_tsx()))
    except Exception as e:
        log.info("tree-sitter not available (%s); falling back to regex parser for TS", e)
        _TS_PARSER = None
    return _TS_PARSER

_TS_IMPORT_RE = re.compile(
    r"""(?:^|\s)
        (?:import\s+(?:[^'"\n]+\s+from\s+)?|export\s+\*?\s*(?:\{[^}]+\}\s*)?from\s+)
        ['"](\.{1,2}/[\w@./\-]+|@[\w/\-]+/[\w./\-]+|[\w@./\-]+)['"]
    """,
    re.MULTILINE | re.VERBOSE,
)

# Caps so we don't explode context
_MAX_FOLLOWS_PER_FILE = 6
_MAX_BYTES_PER_FILE = 24_000


def _python_imports(source: str) -> list[str]:
    """Return a list of module-like dotted paths from a Python source.

    For `from a.b import c`, we return ['a.b'] (and try to find a/b.py).
    Bare `import a.b` returns ['a.b'].
    Relative imports return e.g. '.foo' or '..bar' for downstream handling.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            prefix = "." * node.level
            if not mod and prefix:
                out.append(prefix)
            elif mod:
                out.append(f"{prefix}{mod}")
    return out


def _ts_imports(source: str) -> list[str]:
    """Return relative import specifiers from TS/JS source.

    Tries tree-sitter first for accuracy (handles dynamic imports, re-exports,
    multi-line forms). Falls back to regex if tree-sitter isn't installed."""
    parser = _get_ts_parser()
    if parser is not None:
        return _ts_imports_tree_sitter(source, parser)
    return _ts_imports_regex(source)


def _ts_imports_tree_sitter(source: str, parser) -> list[str]:
    """Walk the AST collecting import/export source strings. Only keeps
    relative paths (./, ../)."""
    try:
        tree = parser.parse(source.encode("utf-8", errors="replace"))
    except Exception as e:
        log.warning("tree-sitter parse failed, falling back to regex: %s", e)
        return _ts_imports_regex(source)
    out: list[str] = []

    def walk(node):
        # Nodes we care about: import_statement, export_statement,
        # and call_expression for dynamic `import('./x')`.
        t = node.type
        if t in ("import_statement", "export_statement"):
            for child in node.children:
                if child.type == "string":
                    spec = _string_node_text(child)
                    if spec and spec.startswith("."):
                        out.append(spec)
        elif t == "call_expression":
            # dynamic import('...')
            head = node.children[0] if node.children else None
            if head is not None and head.type == "import":
                for child in node.children:
                    if child.type == "arguments":
                        for arg in child.children:
                            if arg.type == "string":
                                spec = _string_node_text(arg)
                                if spec and spec.startswith("."):
                                    out.append(spec)
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    # Dedupe preserving order.
    seen: set[str] = set()
    deduped: list[str] = []
    for s in out:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped


def _string_node_text(node) -> str | None:
    """Extract the text between quotes from a tree-sitter `string` node."""
    for child in node.children:
        if child.type == "string_fragment":
            try:
                return child.text.decode("utf-8", errors="replace")
            except (UnicodeError, AttributeError):
                return None
    return None


def _ts_imports_regex(source: str) -> list[str]:
    """Regex fallback when tree-sitter isn't available."""
    out: list[str] = []
    for m in _TS_IMPORT_RE.finditer(source):
        spec = m.group(1)
        if spec.startswith("."):
            out.append(spec)
    return out


def _resolve_python(
    module_path: str, current_file: Path, repo_root: Path
) -> Path | None:
    """Resolve a Python `module_path` (dotted) to an actual .py file inside repo_root."""
    # Relative imports: count leading dots, walk up that many dirs from current_file.
    leading_dots = len(module_path) - len(module_path.lstrip("."))
    if leading_dots:
        rest = module_path[leading_dots:].split(".")
        base = current_file.parent
        for _ in range(leading_dots - 1):
            base = base.parent
        candidates = [
            base.joinpath(*rest).with_suffix(".py"),
            base.joinpath(*rest, "__init__.py"),
        ]
    else:
        parts = module_path.split(".")
        # Try inside the same Python "source root" (i.e., the dir containing the
        # nearest ancestor with `__init__.py` or `pyproject.toml`/`setup.py`).
        # Practical heuristic: try walking up from current_file looking for matches.
        candidates = []
        ancestor = current_file.parent
        while True:
            candidates.append(ancestor.joinpath(*parts).with_suffix(".py"))
            candidates.append(ancestor.joinpath(*parts, "__init__.py"))
            if ancestor == repo_root or ancestor.parent == ancestor:
                break
            ancestor = ancestor.parent
        # Also: from a `<repo>/src/<thing>` layout, try repo_root/src/<parts>.
        if (repo_root / "src").is_dir():
            candidates.append(repo_root / "src" / Path(*parts).with_suffix(".py"))
            candidates.append(repo_root / "src" / Path(*parts) / "__init__.py")

    for c in candidates:
        try:
            c_resolved = c.resolve()
        except OSError:
            continue
        # Must remain inside repo_root
        try:
            c_resolved.relative_to(repo_root.resolve())
        except ValueError:
            continue
        if c_resolved.is_file():
            return c_resolved
    return None


def _resolve_ts(spec: str, current_file: Path, repo_root: Path) -> Path | None:
    """Resolve a relative TS/JS import spec from current_file's directory.
    Tries the standard extensions and `index.ts` fallback."""
    target = (current_file.parent / spec).resolve()
    try:
        target.relative_to(repo_root.resolve())
    except ValueError:
        return None
    exts = [".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"]
    if target.is_file():
        return target
    for ext in exts:
        p = target.with_suffix(ext)
        if p.is_file():
            return p
    # index fallback
    for ext in exts:
        idx = target / f"index{ext}"
        if idx.is_file():
            return idx
    return None


def follow_imports(file_path: Path, repo_root: Path) -> list[Path]:
    """Return a deduped list of file paths reachable in one import hop.
    Conservative caps applied: up to _MAX_FOLLOWS_PER_FILE results."""
    suffix = file_path.suffix.lower()
    try:
        source = file_path.read_bytes()[: _MAX_BYTES_PER_FILE * 2].decode("utf-8", errors="replace")
    except OSError:
        return []

    raw_imports: list[str]
    resolve_fn = None
    if suffix == ".py":
        raw_imports = _python_imports(source)
        resolve_fn = _resolve_python
    elif suffix in (".ts", ".tsx", ".js", ".jsx"):
        raw_imports = _ts_imports(source)
        resolve_fn = _resolve_ts
    else:
        return []

    out: list[Path] = []
    seen: set[Path] = set()
    for imp in raw_imports:
        if len(out) >= _MAX_FOLLOWS_PER_FILE:
            break
        target = resolve_fn(imp, file_path, repo_root)
        if target and target not in seen and target != file_path:
            seen.add(target)
            out.append(target)
    return out
