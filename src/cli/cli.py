from fire import Fire 
from src.indexing.indexing_pipeline import IndexingPipeline
from src.retrieving.retriever_pipeline import RetrieverPipeline
from src.message.success import SuccessMessage
from src.models.models import MinimalSource
from src.utils.default_path import DefaultPath
from src.answer.answer_pipeline import AnswerPipeline

class Flags():
    @staticmethod
    def index(
        max_chunk_size: int = 2000,
        repository_path: str = 'data/raw/vllm-0.10.1',
        chunk_overlap: int = 0,
    ):
        indexer = IndexingPipeline(max_chunk_size, chunk_overlap)
        indexer.browse_raw_for_chunking(repository_path)
        indexer.indexing()
        print(SuccessMessage(f"Ingestion complete! Indices saved under {DefaultPath.output}"))

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
        ):
        retriever = RetrieverPipeline(k)
        retriever.retrieve_chunks_for_dataset(dataset_path, save_directory)
        print(SuccessMessage((
                f"Saved student_search_results to {save_directory}"
            )))

    def answer(self, query: str, k: int):
        retrieved_source = self.search(query, k)
        
        answer = AnswerPipeline().answer_generating(query, retrieved_source)
        print(answer)

    @staticmethod
    def answer_dataset(student_search_results_path: str, save_directory: str):
        test = AnswerPipeline().answer_generating_with_queries(student_search_results_path, save_directory)
        print(SuccessMessage(
            "Saved student_search_results_and_answer to "
            f"{save_directory}"
            )
        )

    @staticmethod
    def evaluate(student_search_results_path: str, dataset_path: str):
        # wait implementation of evaluate methodes
        print(student_search_results_path, dataset_path)


class FlagManagement:
    def flags_retrieve(self):
        self.flags_datas = Fire(Flags)
