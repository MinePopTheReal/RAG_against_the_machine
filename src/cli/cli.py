from src.retrieving.retriever_pipeline import RetrieverPipeline
from src.indexing.indexing_pipeline import IndexingPipeline
from src.answer.answer_pipeline import AnswerPipeline
from src.evaluation.evaluation import Evaluation
from src.utils.default_path import DefaultPath
from src.message.success import SuccessMessage
from src.models.models import MinimalSource
from pydantic import ValidationError
from src.models.check_input import (
    CheckIndex,
    CheckSearch,
    CheckSearchDataset,
    CheckAnswer,
    CheckAnswerDataset,
    CheckEvaluate
)
from src.message.errors import (
    IndexingError,
    RetrievingError,
    AnswerError,
    EvaluateError,
    PydanticError
)
from typing import Any, Literal
from typing import Any


class Flags:
    @staticmethod
    def index(
        max_chunk_size: int = 2000,
        repository_path: str = 'data/raw/vllm-0.10.1',
        chunk_overlap: int = 200,
        mode: Literal["cuda", "cpu", "bm25-only", "default"] = "bm25-only"
    ) -> None:
        print(chunk_overlap, type(chunk_overlap))
        try:
            valid_input = CheckIndex(
                max_chunk_size=max_chunk_size,
                repository_path=repository_path,
                chunk_overlap=chunk_overlap,
                mode=mode
            )
        except ValidationError as e:
            raise PydanticError("Bad input", e, IndexingError)
        indexer = IndexingPipeline(
            valid_input.max_chunk_size, 
            valid_input.chunk_overlap, 
            valid_input.mode
        )
        indexer.browse_raw_for_chunking(valid_input.repository_path)
        indexer.indexing()
        print(SuccessMessage(
            f"Ingestion complete! Indexs saved under {DefaultPath.output}"
        ))

    @staticmethod
    def search(
        query: str, 
        k: int = 5, 
    ) -> list[MinimalSource]:
        try:
            valid_input = CheckSearch(query=query, k=k)
        except ValidationError as e:
            raise PydanticError("Bad input", e, RetrievingError)
        retriever = RetrieverPipeline(valid_input.k)
        retrieved_chunk = retriever.retrieve_chunks_for_query(valid_input.query)

        return retrieved_chunk

    @staticmethod
    def search_dataset(
        dataset_path: str,
        save_directory: str,
        k: int = 5,
    ) -> None:
        try:
            valid_input = CheckSearchDataset(
                dataset_path=dataset_path,
                save_directory=save_directory,
                k=k
                )
        except ValidationError as e:
            raise PydanticError("Bad input", e, RetrievingError)
        retriever = RetrieverPipeline(valid_input.k)
        save_file_path = retriever.retrieve_chunks_for_dataset(
            valid_input.dataset_path,
            valid_input.save_directory
        )
        print(SuccessMessage((
                f"Saved student_search_results to {save_file_path}"
            )))

    def answer(self, query: str, k: int) -> Any:
        try:
            valid_input = CheckAnswer(query=query, k=k)
        except ValidationError as e:
            raise PydanticError("Bad input", e, AnswerError)
        retrieved_source = self.search(valid_input.query, valid_input.k)

        answer = AnswerPipeline()
        result = answer.answer_generating_for_query(query, retrieved_source)
        return result

    @staticmethod
    def answer_dataset(
        student_search_results_path: str,
        save_directory: str
    ) -> None:
        try:
            valid_input = CheckAnswerDataset(
                student_search_results_path=student_search_results_path,
                save_directory=save_directory
                )
        except ValidationError as e:
            raise PydanticError("Bad input", e, AnswerError)
        answer_pipeline = AnswerPipeline()

        answer_pipeline.answer_generating_for_dataset(
            valid_input.student_search_results_path,
            valid_input.save_directory
        )

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
        try:
            valid_input = CheckEvaluate(
                student_search_results_path=student_search_results_path,
                dataset_path=dataset_path
                )
        except ValidationError as e:
            raise PydanticError("Bad input", e, EvaluateError)
        evaluation = Evaluation(
            valid_input.student_search_results_path, 
            valid_input.dataset_path
        )

        evaluation.display_evaluation()
