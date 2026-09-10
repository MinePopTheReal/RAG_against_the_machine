from faiss import write_index, IndexFlatIP, Index
from src.utils.default_path import DefaultPath
from sentence_transformers import (
    SentenceTransformer,
    export_static_quantized_openvino_model
)
from abc import ABC, abstractmethod
from bm25s import tokenize, BM25
from typing import Any
from pathlib import Path
import transformers


transformers.logging.disable_progress_bar()


class Indexer(ABC):
    def __init__(self, save_path: str):
        self.save_path: str = save_path

    @abstractmethod
    def _create_index(self, corpus: list[str]) -> Any:
        ...

    @abstractmethod
    def _save_index(self, index: Any) -> None:
        ...

    def create_and_save_index(self, corpus: list[str]) -> None:
        index = self._create_index(corpus)
        self._save_index(index)


class IndexerBm25(Indexer):
    def _create_index(self, corpus: list[str]) -> BM25:
        corpus_tokens = tokenize(
            corpus,
            show_progress=True
            )
        retriever = BM25(corpus=corpus)
        retriever.index(corpus_tokens)

        return retriever

    def _save_index(self, index: BM25) -> None:
        index.save(self.save_path)


def get_model(
    device: str,
    model_name: str = "BAAI/bge-small-en-v1.5",
    quantized_file_name: str = "openvino_model_qint8_quantized.xml",
    export_path: str = "data/processed/bge-small-openvino"
    ) -> Any:

    if device == "cpu":
        export_dir = Path(export_path)
        config_file = Path(export_dir / "openvino") / "config.json"

        if not (Path(Path(export_dir) / f"openvino/{quantized_file_name}").exists() and config_file.exists()):
            base_model = SentenceTransformer(model_name, backend="openvino")
            base_model.save_pretrained(str(Path(export_dir) / "openvino"))
            export_static_quantized_openvino_model(
                base_model,
                quantization_config=None,
                model_name_or_path=str(export_dir),
            )
        else:
            pass

        return SentenceTransformer(
            str(Path(export_dir) / "openvino"),
            backend="openvino",
            model_kwargs={"file_name": quantized_file_name},
        )
    else:
        return SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)


class IndexerSemanticEmbedding(Indexer):
    def __init__(
        self,
        device: str,
        save_path: str = DefaultPath.semantic_index,
    ):
        self.device = device
        super().__init__(save_path)

    def _create_index(self, corpus: list[str]) -> IndexFlatIP:
        batch_size = 4 if self.device == "cpu" else 256

        model = get_model(self.device)

        vector = model.encode(
            corpus,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=True
        )

        index = IndexFlatIP(vector.shape[1])
        index.add(vector)

        return index

    def _save_index(self, index: Index) -> None:
        write_index(index, str(self.save_path))
