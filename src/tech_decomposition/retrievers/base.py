from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import EnrichedQuery, RepoContext


class Retriever(ABC):
    name: str = "base"

    @abstractmethod
    async def retrieve(self, repo: str, query: EnrichedQuery) -> RepoContext: ...
