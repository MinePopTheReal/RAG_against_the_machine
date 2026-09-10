from src.retrieving.retriever import BM25Retrieving, EmbeddingRetrieving
from concurrent.futures import ThreadPoolExecutor
from src.utils.default_path import DefaultPath
from src.utils.file_manager import FileManager
from src.retrieving.re_rank import ReRank
from src.models.models import (
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    AnsweredQuestion,
    UnansweredQuestion
)
from pathlib import Path
from torch import cuda
from json import loads
from tqdm import tqdm
from logging import getLogger, ERROR


getLogger("huggingface_hub").setLevel(ERROR)


class RetrieverPipeline:
    def __init__(self, k: int) -> None:
        self.k = k

        self.device = "cuda" if cuda.is_available() else "cpu"

        self._executor = ThreadPoolExecutor(max_workers=2)

        self.minimal_source: list[MinimalSource] = [
            MinimalSource(**source)
            for source in loads(
            FileManager.read(DefaultPath.minimal_source)
            )
        ]

        self.bm25_retriever = BM25Retrieving(DefaultPath.bm25_index)
        self.embedding_retriever = EmbeddingRetrieving(DefaultPath.semantic_index, self.device)
        self.reranker = ReRank(self.device)

    def load_batches(self, queries: list[str] | str) -> tuple[list[list[int]], list[list[int]]]:
        bm25_batches = self._executor.submit(
            self.bm25_retriever._retrieving,
            queries,
            self.k
        )

        embedding_batches = self._executor.submit(
            self.embedding_retriever._retrieving,
            queries,
            self.k
        )
        
        return (bm25_batches.result(), embedding_batches.result())

    @staticmethod
    def _load_datasets(datasets_file_path: str) -> list[AnsweredQuestion | UnansweredQuestion]:
        data = FileManager().load(datasets_file_path, RagDataset)

        return data.rag_questions

    def retrieve_chunks_for_query(self, query: str) -> list[MinimalSource]:
        bm25_batches, embedding_batches = self.load_batches(query)
        combined = list(set(bm25_batches[0] + embedding_batches[0]))
        result = self.reranker.re_ranking(query, combined, self.k)

        return [self.minimal_source[idx] for idx in result]

    def retrieve_chunks_for_dataset(self, dataset_path: str, save_directory: str) -> str:
        datas = self._load_datasets(dataset_path)
        queries = [d.question for d in datas]

        bm25_batches, embedding_batches = self.load_batches(queries)

        search_results = []
        for data, bm25_idxs, emb_idxs in tqdm(
            zip(datas, bm25_batches, embedding_batches),
            total=len(datas),
            desc=f"{"Retriving":<15.15}",
            colour="cyan",
            unit="queries",
            ascii="·■"
        ):
            combined = list(set(bm25_idxs + emb_idxs))
            reranked = self.reranker.re_ranking(data.question, combined, self.k)
            search_results.append(MinimalSearchResults(
                question_id=data.question_id,
                question=data.question,
                retrieved_sources=[self.minimal_source[idx] for idx in reranked]
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
        FileManager.write({"search_results":save_result, "k":self.k}, path_save_file)

        return (path_save_file)
