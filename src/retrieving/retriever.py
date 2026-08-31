from abc import ABC, abstractmethod
from bm25s import BM25, tokenize
from faiss import read_index
from src.utils.default_path import DefaultPath
from numpy import array
from src.message.errors import RetrievingError
from src.models.models import MinimalSource
from typing import Any

class Retriver(ABC):
    def __init__(self, save_path: str, minimal_source: list[MinimalSource]):
        self.save_path: str = save_path
        self.minimal_source: list[MinimalSource] = minimal_source

    @abstractmethod
    def _load(self) -> Any:
        ...

    @abstractmethod
    def _retrieving(self) -> list[MinimalSource]:
        ...


class BM25Retrieving(Retriver):
    def _load(self):
        retriever = BM25.load(self.save_path)

        return retriever

    def _retrieving(self, query, k):
        retriever = self._load()

        minimal_source = self._load_index_json(DefaultPath.minimal_source)
        loaded_minimal_source = array([minimal_source])
        query_tokens = tokenize(query)

        docs, scores = retriever.retrieve(query_tokens, k=k)

        docs = [doc for i, doc in enumerate(docs) if scores[0][i]]
        retieval_minimal_source = []

        if not docs:
            raise RetrievingError("No chunks were found.")

        for doc in docs[0]:
            minimal_source = loaded_minimal_source[0][doc]
            retieval_minimal_source.append(MinimalSource(**minimal_source))

        return retieval_minimal_source

class EmbeddingRetrieving(Retriver):
    def _load(self):
        semantic_index = read_index(self.save_path)

        return semantic_index

    def _retrieving(self):
        retriever = self._load()


