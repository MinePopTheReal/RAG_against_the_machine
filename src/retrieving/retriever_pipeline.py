from src.retrieving.retriever import BM25Retrieving, EmbeddingRetrieving
from src.message.errors import PydanticError, RetrievingError
from concurrent.futures import ThreadPoolExecutor
from src.utils.default_path import DefaultPath
from src.utils.file_manager import FileManager
from src.models.check_input import ModeModel
from src.retrieving.re_rank import ReRank
from logging import getLogger, ERROR
from pydantic import ValidationError
from src.models.models import (
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    AnsweredQuestion,
    UnansweredQuestion
)
from pathlib import Path
from json import loads
from tqdm import tqdm

getLogger("huggingface_hub").setLevel(ERROR)


class RetrieverPipeline:
    """
    retrieve pipeline
    """
    def __init__(self, k: int) -> None:
        self.k = k

        self.mode = ModeModel()
        self.mode.load_mode()

        self._executor = ThreadPoolExecutor(max_workers=2)

        try:
            self.minimal_source: list[MinimalSource] = [
                MinimalSource(**source)
                for source in loads(
                    FileManager.read(DefaultPath.minimal_source)
                )
            ]
        except ValidationError as e:
            raise PydanticError("Invalid file content", e, RetrievingError)

        self.bm25_retriever = BM25Retrieving(DefaultPath.bm25_index)

        if self.mode.can_embed:
            self.embedding_retriever = EmbeddingRetrieving(
                DefaultPath.semantic_index,
                self.mode
            )
            self.reranker = ReRank(self.mode)

    def load_batches(
        self,
        queries: list[str] | str
    ) -> tuple[list[list[int]], list[list[int]] | None]:
        bm25_batches = self._executor.submit(
            self.bm25_retriever._retrieving,
            queries,
            self.k
        ).result()

        if self.mode.can_embed:
            embedding_batches = self._executor.submit(
                self.embedding_retriever._retrieving,
                queries,
                self.k
            ).result()
        else:
            embedding_batches = None

        return (bm25_batches, embedding_batches)

    @staticmethod
    def _load_datasets(
        datasets_file_path: str
    ) -> list[AnsweredQuestion | UnansweredQuestion]:
        data = FileManager().load(datasets_file_path, RagDataset)

        return data.rag_questions

    def retrieve_chunks_for_query(self, query: str) -> list[MinimalSource]:
        bm25_batches, embedding_batches = self.load_batches(query)

        if self.mode.can_embed and embedding_batches:
            combined = list(set(bm25_batches[0] + embedding_batches[0]))
            result = self.reranker.re_ranking(query, combined, self.k)
        else:
            result = bm25_batches[0]

        return [self.minimal_source[idx] for idx in result]

    def retrieve_chunks_for_dataset(
        self,
        dataset_path: str,
        save_directory: str
    ) -> str:
        datas = self._load_datasets(dataset_path)
        queries = [d.question for d in datas]

        bm25_batches, embedding_batches = self.load_batches(queries)

        search_results = []
        if not self.mode.can_embed or not embedding_batches:
            for data, bm25_idx in zip(datas, bm25_batches):
                search_results.append(MinimalSearchResults(
                    question_id=data.question_id,
                    question=data.question,
                    retrieved_sources=[
                        self.minimal_source[idx]
                        for idx in bm25_idx
                    ]
                ))
        else:
            for data, bm25_idxs, emb_idxs in tqdm(
                zip(datas, bm25_batches, embedding_batches),
                total=len(datas),
                desc=f"{"Retriving":<15.15}",
                colour="cyan",
                unit="queries",
                ascii="·■"
            ):
                combined = list(set(bm25_idxs + emb_idxs))
                reranked = self.reranker.re_ranking(
                    data.question,
                    combined,
                    self.k
                )
                search_results.append(MinimalSearchResults(
                    question_id=data.question_id,
                    question=data.question,
                    retrieved_sources=[
                        self.minimal_source[idx]
                        for idx in reranked
                    ]
                ))

        save_result = []
        for result in search_results:
            save_result.append(
                {
                    "question_id": result.question_id,
                    "question": result.question,
                    "retrieved_sources": [{
                        "file_path": source.file_path,
                        "first_character_index": source.first_character_index,
                        "last_character_index": source.last_character_index
                        } for source in result.retrieved_sources]
                }
            )

        path_save_file = str(Path(save_directory) / Path(dataset_path).name)
        FileManager.write(
            {
                "search_results": save_result,
                "k": self.k
            },
            path_save_file
        )

        return (path_save_file)
