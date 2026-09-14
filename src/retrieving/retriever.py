from src.message.errors import RetrievingError
from src.indexing.indexer import get_model
from faiss import read_index, Index
from abc import ABC, abstractmethod
from bm25s import BM25, tokenize
from typing import Any
from src.models.check_input import ModeModel


class Retriver(ABC):
    def __init__(self, save_path: str):
        self.save_path: str = save_path

        self.retriever = self._load()

    @abstractmethod
    def _load(self) -> Any:
        ...

    @abstractmethod
    def _retrieving(self, queries: list[str], k: int) -> list[list[int]]:
        ...


class BM25Retrieving(Retriver):
    def _load(self) -> BM25:
        retriever = BM25.load(self.save_path)

        return retriever

    def _retrieving(self, queries: list[str] | str, k: int) -> list[list[int]]:
        queries_tokens = tokenize(queries)

        docs, scores = self.retriever.retrieve(queries_tokens, k=k)

        results = [
            row[score > 0].tolist()
            for row, score in zip(docs, scores)
        ]

        if not any(results):
            raise RetrievingError("No chunks were found.")
        return results


class EmbeddingRetrieving(Retriver):
    def __init__(self, save_path: str, device: ModeModel) -> None:
        super().__init__(save_path)
        self.model = get_model(device)

    def _load(self) -> Index:
        semantic_index = read_index(str(self.save_path))

        return semantic_index

    def _retrieving(self, queries: list[str] | str, k: int) -> list[list[int]]:
        if isinstance(queries, str):
            queries = [queries]
        queries_vector = self.model.encode(queries, normalize_embeddings=True)
        _, indexs = self.retriever.search(queries_vector, k)

        return [
            indexs[i][indexs[i] != -1].tolist()
            for i in range(indexs.shape[0])
            ]
