from bm25s import BM25, tokenize
from pathlib import Path
from src.utils.file_manager import FileManager
from json import loads
from src.models.models import MinimalSearchResults, MinimalSource
import numpy as np
from src.utils.default_path import DefaultPath
from tqdm import tqdm


class RetrieverPipeline:
    def __init__(self, k: int) -> None:
        self.k = k


    @staticmethod
    def _load_index()-> BM25:
        retriever_loaded = BM25.load(DefaultPath.index)
        return retriever_loaded

    @staticmethod
    def _load_index_json(minimal_source_file_path: str):
        minimal_source = FileManager._read(minimal_source_file_path)

        return loads(minimal_source)

    @staticmethod
    def _load_datasets(datasets_file_path: str) -> list[dict[str, str]]:
        data = FileManager._read(datasets_file_path)

        data_obj = loads(data)
        return data_obj['rag_questions']

    def retrieve_chunks_for_query(self, query: str) -> MinimalSource:
        retriever = self._load_index()
        minimal_source = self._load_index_json(DefaultPath.minimal_source)

        loaded_minimal_source = np.array([minimal_source])

        query_tokens = tokenize(query)

        docs, scores = retriever.retrieve(query_tokens, k=self.k)

        docs = [doc for i, doc in enumerate(docs) if scores[0][i] >= 0.5]

        retieval_minimal_source = []
        for doc in docs[0]:
            minimal_source = loaded_minimal_source[0][doc]
            retieval_minimal_source.append(MinimalSource(**minimal_source))

        return retieval_minimal_source


    def retrieve_chunks_for_dataset(self, dataset_path, save_directory):
        search_results: list[MinimalSearchResults] = []

        datas = self._load_datasets(dataset_path)

        for data in tqdm(
            datas,
            desc=f"{"Retriving":<15.15}",
            colour="cyan",
            unit="queries",
            ascii="·■"
        ):
            retieval_minimal_source = self.retrieve_chunks_for_query(data["question"])

            search_results.append(
                MinimalSearchResults(
                    question_id=data['question_id'],
                    question=data['question'],
                    retrieved_sources=retieval_minimal_source
                    )
                )

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
                        
                    }for source in result.retrieved_sources]
                }
            )

        FileManager.write(save_result, save_directory)

