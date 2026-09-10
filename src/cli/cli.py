from src.retrieving.retriever_pipeline import RetrieverPipeline
from src.indexing.indexing_pipeline import IndexingPipeline
from src.answer.answer_pipeline import AnswerPipeline
from src.evaluation.evaluation import Evaluation
from src.utils.default_path import DefaultPath
from src.message.success import SuccessMessage
from src.models.models import MinimalSource
from typing import Any


class Flags:
    @staticmethod
    def index(
        max_chunk_size: int = 2000,
        repository_path: str = 'data/raw/vllm-0.10.1',
        chunk_overlap: int = 200,
    ) -> None:
        indexer = IndexingPipeline(max_chunk_size, chunk_overlap)
        indexer.browse_raw_for_chunking(repository_path)
        indexer.indexing()
        print(SuccessMessage(
            f"Ingestion complete! Indexs saved under {DefaultPath.output}"
        ))

    @staticmethod
    def search(query: str, k: int = 5) -> list[MinimalSource]:
        retriever = RetrieverPipeline(k)
        retrieved_chunk = retriever.retrieve_chunks_for_query(query)

        return retrieved_chunk

    @staticmethod
    def search_dataset(
        dataset_path: str,
        save_directory: str,
        k: int = 5
    ) -> None:
        retriever = RetrieverPipeline(k)
        save_file_path = retriever.retrieve_chunks_for_dataset(
            dataset_path,
            save_directory
        )
        print(SuccessMessage((
                f"Saved student_search_results to {save_file_path}"
            )))

    def answer(self, query: str, k: int) -> Any:
        retrieved_source = self.search(query, k)

        answer = AnswerPipeline()
        result = answer.answer_generating_for_query(query, retrieved_source)
        return result

    @staticmethod
    def answer_dataset(
        student_search_results_path: str,
        save_directory: str
    ) -> None:
        answer_pipeline = AnswerPipeline()

        answer_pipeline.answer_generating_for_dataset(
            student_search_results_path,
            save_directory
        )
        print(SuccessMessage(
            "Saved student_search_results_and_answer to "
            f"{save_directory}"
            )
        )

    @staticmethod
    def evaluate(student_search_results_path: str, dataset_path: str) -> None:
        evaluation = Evaluation(student_search_results_path, dataset_path)

        evaluation.display_evaluation()
