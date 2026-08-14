from src.message.success import SuccessMessage
from src.indexing.chunker import Chunker, MinimalSource
from os import walk, path, scandir
from pathlib import Path
from src.indexing.Indexer import Indexer
from tqdm import tqdm
from src.utils.file_manager import FileManager

class IndexingPipeline:
    def __init__(
        self, 
        chunk_size: int, 
        chunk_overlap: int, 
        output_folder_path: str
    ):
        self.output_folder_path: str = output_folder_path

        self.chunk_size: int = chunk_size
        self.chunk_overlap: int = chunk_overlap
        
        self.chunker = Chunker(self.chunk_size, self.chunk_overlap)
        self.indexer = Indexer()

        self.chunks: list[MinimalSource] = []


    @property
    def create_corpus(self):
        return [chunk.text for chunk in self.chunks]

    def browse_raw_for_chunking(
        self,
        root_path_of_data: str,
    ) -> None:
        valid_extensions = {"py", "txt", "md", ""}

        for current_path, _, files in tqdm(
            list(walk(root_path_of_data)),
            desc=f"{"Chunking":<15.15}",
            colour="cyan",
            unit="folder",
            ascii="·■"
        ):
            files_pbar = tqdm(
                files,
                desc="Files",
                colour="cyan",
                leave=False,
                unit="file",
                bar_format="{l_bar}{bar:30}{r_bar}",
                ascii="·■"
            )

            for file_name in files_pbar:
                files_pbar.set_description(
                    f"File: {file_name:<30.30}"
                )

                extension = Path(file_name).suffix.removeprefix(".")

                if extension not in valid_extensions:
                    continue

                file_path = Path(current_path) / file_name
                file_data = FileManager._read(file_path)

                self.chunks.extend(
                    self.chunker.chunker(
                        file_data,
                        str(file_path),
                        extension,
                    )
                )
        self.chunker.output(self.chunks, Path(self.output_folder_path) / "chunks.json")

    def indexing(self):
        corpus = self.create_corpus
        retriever = self.indexer.bm25(corpus)
        retriever.save(Path(self.output_folder_path) / "bm25")




