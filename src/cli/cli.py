from fire import Fire 
from src.indexing.indexing_pipeline import IndexingPipeline
from src.retrieving.retriever_pipeline import RetrieverPipeline
from src.message.success import SuccessMessage


class Flags():
    @staticmethod
    def index(
        max_chunck_size: int = 2000,
        repository_path: str = 'data/raw/vllm-0.10.1',
        chunk_overlap: int = 0,
        output_path: str = 'data/processed'
    ):
        indexer = IndexingPipeline(max_chunck_size, chunk_overlap, output_path)
        indexer.browse_raw_for_chunking(repository_path)
        indexer.indexing()
        print(SuccessMessage(f"Ingestion complete! Indices saved under {output_path}"))

    @staticmethod
    def search(query: str, k: int):
        # wait implementation of search methodes
        print(query, k)

    @staticmethod
    def search_dataset(
        datasets_path: str,
        save_directory: str,
        k: int = 5
        ):
        retriever = RetrieverPipeline(datasets_path, k, save_directory)
        # wait implementation of search_dataset methodes
        print(datasets_path, k, save_directory)

    @staticmethod
    def answer(query: str, k: int):
        # wait implementation of answer methodes
        print(query, k)

    @staticmethod
    def answer_dataset(student_search_results_path: str, save_directory: str):
        # wait implementation of answer_dataset methodes
        print(student_search_results_path, save_directory)

    @staticmethod
    def evaluate(student_search_results_path: str, dataset_path: str):
        # wait implementation of evaluate methodes
        print(student_search_results_path, dataset_path)


class FlagManagement:
    def flags_retrieve(self):
        self.flags_datas = Fire(Flags)
