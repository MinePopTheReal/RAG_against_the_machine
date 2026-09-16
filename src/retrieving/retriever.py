from src.message.errors import RetrievingError
from src.indexing.indexer import get_model
from faiss import read_index, Index
from abc import ABC, abstractmethod
from bm25s import BM25, tokenize
from typing import Any
from src.models.check_input import ModeModel
from pathlib import Path


class Retriver(ABC):
    """
    retriever
    """
    def __init__(self, save_path: str):
        self.save_path: str = save_path

        self.retriever = self._load()

    @abstractmethod
    def _load(self) -> Any:
        """load indexer

        Returns:
            Any: indexer
        """
        ...

    @abstractmethod
    def _retrieving(self, queries: list[str], k: int) -> list[list[int]]:
        """
        retrieving

        Args:
            queries (list[str]): the questions for which we
            need to find the chunks
            k (int): the number of chunks

        Returns:
            list[list[int]]: a list of lists containing all the indices
            of the retrieved sources
        """
        ...


class BM25Retrieving(Retriver):
    def _load(self) -> BM25:
        """load indexer

        Returns:
            Any: indexer
        """
        retriever = BM25.load(self.save_path)

        return retriever

    def _retrieving(self, queries: list[str] | str, k: int) -> list[list[int]]:
        """
        retrieving

        Args:
            queries (list[str]): the questions for which we
            need to find the chunks
            k (int): the number of chunks

        Returns:
            list[list[int]]: a list of lists containing all the indices
            of the retrieved sources
        """
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
    def __init__(self, save_path: str, mode: ModeModel) -> None:
        """load indexer

        Returns:
            Any: indexer
        """
        super().__init__(save_path)
        self.model = get_model(mode)

    def _load(self) -> Index:
        if not Path(self.save_path).exists():
            raise RetrievingError(
                "The backup file for the "
                "emedding index does not exist"
            )
        semantic_index = read_index(str(self.save_path))

        return semantic_index

    def _retrieving(self, queries: list[str] | str, k: int) -> list[list[int]]:
        """
        retrieving

        Args:
            queries (list[str]): the questions for which we
            need to find the chunks
            k (int): the number of chunks

        Returns:
            list[list[int]]: a list of lists containing all the indices
            of the retrieved sources
        """
        if isinstance(queries, str):
            queries = [queries]
        queries_vector = self.model.encode(queries, normalize_embeddings=True)
        _, indexs = self.retriever.search(queries_vector, k)

        return [
            indexs[i][indexs[i] != -1].tolist()
            for i in range(indexs.shape[0])
            ]
