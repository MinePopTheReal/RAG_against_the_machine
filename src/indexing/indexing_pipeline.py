from src.indexing.indexer import IndexerBm25, IndexerSemanticEmbedding
from src.indexing.chunker import Chunker, MinimalSource
from src.utils.default_path import DefaultPath
from src.utils.file_manager import FileManager
from src.message.errors import IndexingError
from os import walk, listdir
from pathlib import Path
from tqdm import tqdm
from torch import cuda

class IndexingPipeline:
    def __init__(
        self, 
        chunk_size: int, 
        chunk_overlap: int, 
    ):
        self.chunk_size: int = chunk_size
        self.chunk_overlap: int = chunk_overlap
        
        self.chunker = Chunker(self.chunk_size, self.chunk_overlap)

        self.chunks: list[MinimalSource] = []

        self.corpus: list[str] = []

    def browse_raw_for_chunking(
        self,
        root_path_of_data: str,
    ) -> None:
        valid_extensions = {"py", "txt", "md"}

        if not listdir(root_path_of_data):
            raise(IndexingError("The repository you specified is empty."))

        for current_path, _, files in tqdm(
            list(walk(root_path_of_data)),
            desc=f"{"Chunking":<15.15}",
            colour="cyan",
            unit="folder",
            ascii="·■"
        ):
            for file_name in files:

                extension = Path(file_name).suffix.removeprefix(".")

                if extension not in valid_extensions:
                    continue

                file_path = Path(current_path) / file_name
                file_data = FileManager.read(file_path)

                chunks, texts = self.chunker.chunker(
                    file_data,
                    str(file_path),
                    extension,
                )

                self.corpus.extend(texts)
                self.chunks.extend(chunks)
        self.chunker.output(self.chunks, self.corpus)

    def indexing(self) -> None:
        device = "cuda" if cuda.is_available() else "cpu"

        indexer_bm25 = IndexerBm25(DefaultPath.bm25_index)
        indexer_bm25.create_and_save_index(self.corpus)

        indexer_semanctic = IndexerSemanticEmbedding(
            device,
            DefaultPath.semantic_index
            )
        indexer_semanctic.create_and_save_index(self.corpus)
