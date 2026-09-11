from sentence_transformers import CrossEncoder
from src.utils.file_manager import FileManager
from src.utils.default_path import DefaultPath
from json import loads
from typing import Any
import transformers
from src.models.check_input import ModeModel

transformers.logging.set_verbosity_error()

class ReRank:
    def __init__(self, device: ModeModel) -> None:
        self.device = device
        self.reranker: CrossEncoder = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            device=self.device.get_mode
        )
        self.corpus: list[str] = self._load_corpus()

    def _load_corpus(self) -> list[str] | Any:
        corpus = FileManager.read(DefaultPath.chunked_source)
        return loads(corpus)

    def re_ranking(self, queries: str, chunks_idxs: list[int], k: int) -> list[int]:
        chunks_idxs = list(set(chunks_idxs))
        chunks = [self.corpus[idx] for idx in chunks_idxs]

        results = self.reranker.rank(
            queries,
            chunks,
            device=self.device.get_mode,
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

        return [chunks_idxs[result["corpus_id"]] for result in results[:k]]
