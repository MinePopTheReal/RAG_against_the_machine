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
    CheckEvaluate,
    ModeModel
)
from src.message.errors import (
    IndexingError,
    RetrievingError,
    AnswerError,
    EvaluateError,
    PydanticError
)
from typing import Any, Literal


class CLI:
    """
    All methods available to the user for launching the entire pipeline.
    """
    @staticmethod
    def index(
        max_chunk_size: int = 2000,
        repository_path: str = DefaultPath.repository,
        chunk_overlap: int = 200,
        mode: Literal["cuda", "cpu", "bm25-only", "default"] = "bm25-only"
    ) -> None:
        """
        The chunking and BM25 indexing stages can be used on their own,
        but they can also be combined with semantic embedding-based
        recommendation systems if an NVIDIA GPU is available; however,
        it is still possible to use them with just the.

        Args:
            max_chunk_size (int, optional): the maximum number of
            characters a chunk can contain. Defaults to 2000.

            repository_path (str, optional): the path to the folder that
            the search will be based on. Defaults to DefaultPath.repository.

            chunk_overlap (int, optional): the number of characters that will
            be used to reuse certain passages that
            have already been chunked. Defaults to 200.

            mode Literal['cpu', 'cuda', 'bm25-only', 'default']: The
            mode based on your processing power, we
            recommend CUDA if you have an NVIDIA GPU.. Defaults to "bm25-only".
        """
        try:
            valid_input = CheckIndex(
                max_chunk_size=max_chunk_size,
                repository_path=repository_path,
                chunk_overlap=chunk_overlap,
                mode=ModeModel(mode=mode)
            )
        except ValidationError as e:
            raise PydanticError("Bad input", e, IndexingError)
        valid_input.mode.save_mode()
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
        """
        The method that searches for the best chunks available
        in the indexed folder for a given question
        and returns the top k chunks.

        Args:
            query (str): the question for which you'll find chunks.
            k (int, optional): the number of chunks. Defaults to 5.

        Returns:
            list[MinimalSource]: the list of MinimalSources (chunks with the
            file path and the start and end indices of the chunk in the file)
        """
        try:
            valid_input = CheckSearch(query=query, k=k)
        except ValidationError as e:
            raise PydanticError("Bad input", e, RetrievingError)
        retriever = RetrieverPipeline(valid_input.k)
        retrieved_chunk = retriever.retrieve_chunks_for_query(
            valid_input.query
        )

        return retrieved_chunk

    @staticmethod
    def search_dataset(
        dataset_path: str,
        save_directory: str,
        k: int = 5,
    ) -> None:
        """
        A method that searches for the best available segments
        in the indexed folder for a given question dataset
        and saves the k best segments.

        Args:
            dataset_path (str): the path to the question dataset
            save_directory (str): Select the path to the chunk backup folder
            k (int, optional): the number of chunks. Defaults to 5.
        """
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
        """
        The method that generates an answer to a given question
        and use search function for retrieving.

        Args:
            query (str): the question we are going to answer
            k (int): the number of chunks that will be selected

        Returns:
            Any: return response
        """
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
        """
        The method that generates responses using the previously
        selected chunks to answer the associated question

        Args:
            student_search_results_path (str): the path to the output file
            from the 'search_dataset' function
            save_directory (str): the file where the result will be saved
        """
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

    @staticmethod
    def evaluate(student_search_results_path: str, dataset_path: str) -> None:
        """
        The method that check quality of retrieving

        Args:
            student_search_results_path (str): the file path resulting
            from the retrieval
            dataset_path (str): The path to the dataset from
            which we retrieved the data
        """
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
