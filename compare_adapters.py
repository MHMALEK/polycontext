import asyncio
import json
from pathlib import Path

from tech_decomposition.config import get_settings
from tech_decomposition.adapters.registry import get_adapter
from tech_decomposition.adapters.base import AdapterAskInput

async def main():
    settings = get_settings()
    query = "how upload master data works accross apps?"
    repos = list(settings.repos)
    print(f"Running query: '{query}' on repos: {repos}\n")

    input_data = AdapterAskInput(
        query=query,
        repos=repos,
        ticket_url=None
    )

    adapters_to_test = ["gemini", "gemini-grounded"]
    
    results = {}

    for adapter_name in adapters_to_test:
        print(f"=== Running adapter: {adapter_name} ===")
        adapter = get_adapter(adapter_name, settings)
        
        try:
            result = await adapter.ask(input_data)
            results[adapter_name] = result
            print(f"[{adapter_name}] Done. Tokens In: {getattr(result.metrics, 'tokens_in', 0)}, Tokens Out: {getattr(result.metrics, 'tokens_out', 0)}")
        except Exception as e:
            print(f"[{adapter_name}] Failed: {e}")
            results[adapter_name] = None

    print("\n\n=== COMPARISON REPORT ===")
    
    for name in adapters_to_test:
        res = results.get(name)
        if not res:
            continue
            
        print(f"\n## Adapter: {name}")
        print(f"- Cost: ${getattr(res.metrics, 'cost_usd') or 0.0:.4f}")
        print(f"- Duration: {getattr(res.metrics, 'duration_ms') or 0} ms")
        print(f"- Tokens In: {getattr(res.metrics, 'tokens_in') or 0}")
        print(f"- Tokens Out: {getattr(res.metrics, 'tokens_out') or 0}")
        print(f"- Tool Calls: {getattr(res.metrics, 'tool_calls') or 0}")
        
        # Save full answer to a file
        out_file = f"answer_{name}.md"
        with open(out_file, "w") as f:
            f.write(f"# Query: {query}\n\n## Answer\n\n{res.answer}\n\n## Metrics\n{json.dumps(res.metrics.model_dump() if hasattr(res.metrics, 'model_dump') else {}, indent=2)}")
        print(f"(Full answer saved to {out_file})")

if __name__ == "__main__":
    asyncio.run(main())
