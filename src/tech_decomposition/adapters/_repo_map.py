from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Try to load tree_sitter and languages
try:
    import tree_sitter
    import tree_sitter_python
    import tree_sitter_javascript
    import tree_sitter_typescript
    
    LANG_PYTHON = tree_sitter.Language(tree_sitter_python.language())
    LANG_JS = tree_sitter.Language(tree_sitter_javascript.language())
    LANG_TS = tree_sitter.Language(tree_sitter_typescript.language_typescript())
    LANG_TSX = tree_sitter.Language(tree_sitter_typescript.language_tsx())
    
    PARSER_PYTHON = tree_sitter.Parser(LANG_PYTHON)
    PARSER_JS = tree_sitter.Parser(LANG_JS)
    PARSER_TS = tree_sitter.Parser(LANG_TS)
    PARSER_TSX = tree_sitter.Parser(LANG_TSX)
    
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False


def _get_parser_for_file(path: Path) -> tuple[tree_sitter.Parser, str] | None:
    ext = path.suffix.lower()
    if ext == ".py":
        return PARSER_PYTHON, "python"
    elif ext == ".js" or ext == ".jsx":
        return PARSER_JS, "javascript"
    elif ext == ".ts":
        return PARSER_TS, "typescript"
    elif ext == ".tsx":
        return PARSER_TSX, "tsx"
    return None


def _extract_symbols(node: tree_sitter.Node, lang: str) -> list[str]:
    """Recursively walk the AST and extract class and function signatures."""
    symbols = []
    
    # Types that define classes or functions/methods
    target_types = {
        "python": {"class_definition", "function_definition"},
        "javascript": {"class_declaration", "function_declaration", "method_definition", "arrow_function", "variable_declarator"},
        "typescript": {"class_declaration", "interface_declaration", "function_declaration", "method_definition", "arrow_function", "variable_declarator"},
        "tsx": {"class_declaration", "interface_declaration", "function_declaration", "method_definition", "arrow_function", "variable_declarator"},
    }
    
    def walk(n: tree_sitter.Node, depth: int = 0):
        if n.type in target_types.get(lang, set()):
            # Try to find the name of the class/function
            name_node = n.child_by_field_name("name")
            if name_node and name_node.text:
                symbol_type = "class" if "class" in n.type or "interface" in n.type else "func"
                symbols.append(f"{'  ' * depth}└─ [{symbol_type}] {name_node.text.decode('utf8')}")
                depth += 1
            # For arrow functions or exports where name might be in an identifier node
            elif n.type == "variable_declarator":
                name_node = n.child_by_field_name("name")
                if name_node and name_node.text:
                    symbols.append(f"{'  ' * depth}└─ [var/func] {name_node.text.decode('utf8')}")
                    depth += 1
        
        for child in n.children:
            walk(child, depth)

    walk(node)
    return symbols


def generate_repo_map(repos_root: Path, repos: list[str], max_depth: int = 3) -> str:
    """Generate a lightweight architectural skeleton (folders + top files) of the repos.
    If tree-sitter is available, parses code files to extract class and function names.
    """
    lines = ["Repository Structure (AST Map):"]
    
    for repo in repos:
        repo_path = repos_root / repo
        if not repo_path.is_dir():
            continue
        lines.append(f"{repo}/")
        
        def walk_dir(current_dir: Path, current_depth: int, prefix: str = "  "):
            if current_depth >= max_depth:
                return
            try:
                entries = sorted(current_dir.iterdir(), key=lambda p: (not p.is_dir(), p.name))
            except PermissionError:
                return
            
            for p in entries:
                if p.name.startswith(".") or p.name in {"node_modules", "dist", "build", "venv", "__pycache__"}:
                    continue
                
                if p.is_dir():
                    lines.append(f"{prefix}{p.name}/")
                    walk_dir(p, current_depth + 1, prefix + "  ")
                else:
                    # Append file name
                    lines.append(f"{prefix}{p.name}")
                    
                    # If it's a code file and we have tree-sitter, parse the symbols!
                    if TREE_SITTER_AVAILABLE:
                        parser_info = _get_parser_for_file(p)
                        if parser_info:
                            parser, lang = parser_info
                            try:
                                content = p.read_bytes()
                                tree = parser.parse(content)
                                symbols = _extract_symbols(tree.root_node, lang)
                                for sym in symbols:
                                    lines.append(f"{prefix}  {sym}")
                            except Exception as e:
                                log.debug(f"Failed to parse {p}: {e}")

        walk_dir(repo_path, 0)
        
    return "\n".join(lines)
