from sentence_transformers import CrossEncoder
from src.utils.file_manager import FileManager
from src.utils.default_path import DefaultPath
from json import loads
from typing import Any
import transformers
from src.models.check_input import ModeModel

transformers.logging.set_verbosity_error()


class ReRank:
    """
    allows you to retrieve the k best chunks from among
    the k chunks in the index and the k chunks in the embedding
    """
    def __init__(self, mode: ModeModel) -> None:
        self.mode = mode
        self.reranker: CrossEncoder = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            device=self.mode.get_mode
        )
        self.corpus: list[str] = self._load_corpus()

    def _load_corpus(self) -> Any:
        """laod all chunks (corpus)

        Returns:
            Any: return corpus
        """
        corpus = FileManager.read(DefaultPath.chunked_source)
        return loads(corpus)

    def re_ranking(
        self,
        query: str,
        chunks_idxs: list[int],
        k: int
    ) -> list[int]:
        """
        Searches through the bm25 chunks and semantic embeddings
        to find the k best ones based on the base questiona

        Args:
            queries (str): the question for which we are
            looking for the chunks
            chunks_idxs (list[int]): a list index of chunk
            from bm25 and embedding
            k (int): number of chunks to return

        Returns:
            list[int]: return a list of index of
            retrieved chunk from all chunks
        """
        chunks_idxs = list(set(chunks_idxs))
        chunks = [self.corpus[idx] for idx in chunks_idxs]

        results = self.reranker.rank(
            query,
            chunks,
            device=self.mode.get_mode,
            batch_size=256
        )

        ranked_idxs: list[int] = []

        for result in results[:k]:
            corpus_id = result["corpus_id"]

            if not isinstance(corpus_id, int):
                raise TypeError(
                    "Expected corpus_id to be int, got",
                    f"{type(corpus_id).__name__}"
                )

            ranked_idxs.append(chunks_idxs[corpus_id])

        return ranked_idxs
