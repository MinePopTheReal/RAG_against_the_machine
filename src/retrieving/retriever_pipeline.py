from bm25s import BM25, tokenize
from pathlib import Path
from src.utils.file_manager import FileManager
from json import loads
from src.models.models import RagDataset


class RetrieverPipeline:
    def __init__(self, datasets_path: str, k: int, save_directory: str) -> None:
        self.datasets_path = datasets_path
        self.k = k
        self.save_directory = save_directory
        self.k = k

    def _load_index(self) -> BM25:
        retriever_loaded = BM25.save(Path(self.output_path) / "bm25_index")
        return retriever_loaded

    def _load_datasets(self, datasets_file_path):
        data = FileManager._read(datasets_file_path)
        
        data_obj = loads(data)
        print(data_obj)
        return data_obj

    def retrieve(self):
        retriever = self._load_index()
        data = self.load_datasets()
        
        query_tokens = tokenize(data)
        docs, scores = retriever.retrieve(query_tokens, k=self.k)
