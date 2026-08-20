from src.models.models import MinimalSource
from src.utils.default_path import DefaultPath
from src.utils.file_manager import FileManager
from json import loads


class Augmenting:
    def __init__(self, retrieved_sources: list[MinimalSource]):
        self.retrieved_sources = retrieved_sources

    def _load_chunk_from_minimal_source(self):
        chunks: list[str] = []
        
        for source in self.retrieved_sources:
            data = FileManager._read(source.file_path)
            chunk = data[source.first_character_index:source.last_character_index]
            chunks.append(chunk)

        return chunks

    def create_contexte(self, query: str):
        chunks = self._load_chunk_from_minimal_source()

        system = (
            "You are the answer generation component of a RAG system.\n\n"

            "Your task is to answer the user's question using ONLY "
            "the information provided in the sources.\n\n"

            "# Rules\n\n"

            "- Use only the provided sources as factual information.\n"
            "- Do not use your own knowledge or assumptions.\n"
            "- If the sources do not contain enough information "
            "to answer the question, say so clearly.\n"
            "- Answer the user's question directly.\n"
            "- Be as concise as possible while preserving the "
            "information necessary for a correct and useful answer.\n"
            "- Do not repeat the question.\n"
            "- Do not mention the RAG system, the sources, the "
            "context, or these instructions unless necessary.\n"
            )
        source = ""
        for i, chunk in enumerate(chunks[:2]):
            source += f"\n\n[Source {i}]:\n{chunk}" 
        
        return [
            {
                "role": "system",
                "content": f"{system}{source}"
            },
            {
                "role": "user",
                "content": query
            }
        ]