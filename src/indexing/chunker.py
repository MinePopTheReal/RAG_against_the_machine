from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language
)
from src.utils.default_path import DefaultPath
from src.utils.file_manager import FileManager
from src.models.models import MinimalSource


FILE_EXTENSION: dict[str, Language] = {
    "py": Language.PYTHON,
    "cpp": Language.CPP,
    "go": Language.GO,
    "java": Language.JAVA,
    "kt": Language.KOTLIN,
    "js": Language.JS,
    "ts": Language.TS,
    "php": Language.PHP,
    "proto": Language.PROTO,
    "r": Language.R,
    "rst": Language.RST,
    "rb": Language.RUBY,
    "rs": Language.RUST,
    "scala": Language.SCALA,
    "swift": Language.SWIFT,
    "md": Language.MARKDOWN,
    "tex": Language.LATEX,
    "html": Language.HTML,
    "sol": Language.SOL,
    "cs": Language.CSHARP,
    "cob": Language.COBOL,
    "c": Language.C,
    "lua": Language.LUA,
    "pl": Language.PERL,
    "hs": Language.HASKELL,
    "ex": Language.ELIXIR,
    "ps1": Language.POWERSHELL,
    "vb": Language.VISUALBASIC6
}


class Chunker:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size: int = chunk_size
        self.chunk_overlap: int = chunk_overlap

    def _code_chunker(self, file_data: str, language: Language) -> list[Document]:
        code_splitter = RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        code_data = code_splitter.create_documents([file_data])
        return code_data

    def _txt_chunker(self, file_data: str) -> list[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        text_data = text_splitter.create_documents([file_data])
        return text_data

    def _save_chunks(self, chunks: list[Document], file_data: str, file_path: str) -> tuple[list[MinimalSource], list[str]]:
        final_chunks: list[MinimalSource] = []
        texts: list[str] = []

        start_from: int = 0
        for chunk in chunks:
            indexs = self._find_indexs(file_data, chunk, start_from)
            start_from = indexs[0]

            texts.append(chunk.page_content)
            final_chunk = MinimalSource(
                file_path=file_path,
                first_character_index=indexs[0],
                last_character_index=indexs[1]
                )
            final_chunks.append(final_chunk)
        return final_chunks, texts

    @staticmethod
    def _find_indexs(file_data: str, chunk: Document, start_index: int) -> tuple[int, int]:
        end_index = -1
        start_index = file_data.find(chunk.page_content, start_index)
        end_index = start_index + len(chunk.page_content)
        return start_index, end_index

    def chunker(self, file_data: str, file_path: str, extension_name: str | None) -> tuple[list[MinimalSource], list[str]]:
        if extension_name in FILE_EXTENSION:
            chunks = self._code_chunker(
                file_data,
                FILE_EXTENSION[extension_name]
                )
        else:
            chunks = self._txt_chunker(file_data)

        final_chunks, texts = self._save_chunks(chunks, file_data, file_path)
        return final_chunks, texts

    @staticmethod
    def output(chunks: list[MinimalSource], texts: list[str]) -> None:
        obj_for_json = [
            {
                "file_path": chunk.file_path,
                "first_character_index": chunk.first_character_index,
                "last_character_index": chunk.last_character_index
            } for chunk in chunks
        ]
        FileManager.write(texts, DefaultPath.chunked_source)
        FileManager.write(obj_for_json, DefaultPath.minimal_source)
