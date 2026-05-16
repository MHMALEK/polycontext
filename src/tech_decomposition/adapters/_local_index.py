import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

log = logging.getLogger(__name__)

# Try to load tree_sitter and parsers
try:
    from ._repo_map import TREE_SITTER_AVAILABLE, _get_parser_for_file
except ImportError:
    TREE_SITTER_AVAILABLE = False


class LocalSemanticIndex:
    """A local semantic search index powered by ChromaDB and Tree-sitter AST chunking."""
    
    def __init__(self, persist_directory: str = ".chroma_db"):
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory, settings=ChromaSettings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(name="codebase_chunks")
        
    def _extract_chunks(self, node, lang: str, content_bytes: bytes, file_path: str) -> list[dict]:
        """Walk AST and extract full code snippets for classes and functions."""
        chunks = []
        
        target_types = {
            "python": {"class_definition", "function_definition"},
            "javascript": {"class_declaration", "function_declaration", "method_definition"},
            "typescript": {"class_declaration", "interface_declaration", "function_declaration", "method_definition"},
            "tsx": {"class_declaration", "interface_declaration", "function_declaration", "method_definition"},
        }
        
        def walk(n):
            if n.type in target_types.get(lang, set()):
                name_node = n.child_by_field_name("name")
                name = name_node.text.decode('utf8') if name_node and name_node.text else "anonymous"
                
                # Extract the full code of this node
                chunk_code = content_bytes[n.start_byte:n.end_byte].decode('utf8', errors='replace')
                
                chunks.append({
                    "id": f"{file_path}::{name}::{n.start_point[0]}",
                    "text": f"File: {file_path}\nSymbol: {name}\nCode:\n{chunk_code}",
                    "metadata": {
                        "file": file_path,
                        "symbol": name,
                        "start_line": n.start_point[0],
                        "end_line": n.end_point[0],
                    }
                })
            
            # Don't recurse into classes/functions if we already grabbed the whole thing to avoid nested duplicate chunks?
            # Actually, sometimes we want methods inside classes. Let's recurse.
            for child in n.children:
                walk(child)

        walk(node)
        return chunks

    def index_repositories(self, repos_root: Path, repos: list[str]):
        """Parse files, extract chunks, and add to ChromaDB."""
        if not TREE_SITTER_AVAILABLE:
            log.warning("Tree-sitter not available. Local index will be empty.")
            return

        all_documents = []
        all_metadatas = []
        all_ids = []

        for repo in repos:
            repo_path = repos_root / repo
            if not repo_path.is_dir():
                continue
            
            for p in repo_path.rglob("*"):
                if not p.is_file():
                    continue
                # Skip common noise
                if any(part.startswith(".") for part in p.parts) or "node_modules" in p.parts or "__pycache__" in p.parts:
                    continue
                
                parser_info = _get_parser_for_file(p)
                if not parser_info:
                    continue
                
                parser, lang = parser_info
                try:
                    content_bytes = p.read_bytes()
                    tree = parser.parse(content_bytes)
                    chunks = self._extract_chunks(tree.root_node, lang, content_bytes, str(p.relative_to(repos_root)))
                    
                    for chunk in chunks:
                        all_documents.append(chunk["text"])
                        all_metadatas.append(chunk["metadata"])
                        all_ids.append(chunk["id"])
                except Exception as e:
                    log.debug(f"Failed to chunk {p}: {e}")
        
        if all_documents:
            log.info(f"Adding {len(all_documents)} AST chunks to local ChromaDB index...")
            # ChromaDB upsert in batches of 5000 to avoid limits
            batch_size = 5000
            for i in range(0, len(all_documents), batch_size):
                self.collection.upsert(
                    documents=all_documents[i:i+batch_size],
                    metadatas=all_metadatas[i:i+batch_size],
                    ids=all_ids[i:i+batch_size]
                )
            log.info("Local indexing complete.")

    def search(self, query: str, n_results: int = 5) -> list[dict]:
        """Search the local semantic index."""
        if self.collection.count() == 0:
            return []
            
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        snippets = []
        if results and results['documents'] and results['documents'][0]:
            for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                snippets.append({
                    "content": doc,
                    "metadata": meta
                })
        return snippets
