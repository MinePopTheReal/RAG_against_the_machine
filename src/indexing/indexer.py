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
from src.models.check_input import ModeModel


transformers.logging.disable_progress_bar()


class Indexer(ABC):
    """
    manages indexing
    """
    def __init__(self, save_path: str):
        self.save_path: str = save_path

    @abstractmethod
    def _create_index(self, corpus: list[str]) -> Any:
        """
        Create the index

        Args:
            corpus (list[str]): list of all chunks

        Returns:
            Any: return index
        """
        ...

    @abstractmethod
    def _save_index(self, index: Any) -> None:
        """
        Save index

        Args:
            index (Any): the index to save
        """
        ...

    def create_and_save_index(self, corpus: list[str]) -> None:
        """create and save index

        Args:
            corpus (list[str]): list of all chunks
        """
        index = self._create_index(corpus)
        self._save_index(index)


class IndexerBm25(Indexer):
    """
    bm25 indexer
    """
    def _create_index(self, corpus: list[str]) -> BM25:
        """
        Create the index

        Args:
            corpus (list[str]): list of all chunks

        Returns:
            Any: return index
        """
        corpus_tokens = tokenize(
            corpus,
            show_progress=True
            )
        retriever = BM25(corpus=corpus)
        retriever.index(corpus_tokens)

        return retriever

    def _save_index(self, index: BM25) -> None:
        """
        Save index

        Args:
            index (Any): the index to save
        """
        index.save(self.save_path)


def get_model(
    mode: ModeModel,
    model_name: str = "BAAI/bge-small-en-v1.5",
    quantized_file_name: str = "openvino_model_qint8_quantized.xml",
    export_path: str = "data/processed/bge-small-openvino"
) -> Any:
    """
    loads a model if it doesn't already have one
    and adapts based on the selected mode

    Args:
        mode (ModeModel): the choosed mode by user

        model_name (str, optional): the model for semamtic embedding.
        Defaults to "BAAI/bge-small-en-v1.5".

        quantized_file_name (str, optional): the file name for save
        index for semantic embedding. Defaults to
        "openvino_model_qint8_quantized.xml".

        export_path (str, optional): the folder where index will
        be save. Defaults to "data/processed/bge-small-openvino".

    Returns:
        Any: return the loaded model
    """

    if mode.get_mode == "cpu":
        export_dir = Path(export_path)
        config_file = Path(export_dir / "openvino") / "config.json"

        if not (
            Path(Path(export_dir) / f"openvino/{quantized_file_name}").exists()
            and config_file.exists()
        ):
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
        return SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")


class IndexerSemanticEmbedding(Indexer):
    """
    semantic embedding indexer
    """
    def __init__(
        self,
        mode: ModeModel,
        save_path: str = DefaultPath.semantic_index,
    ):
        self.mode = mode
        super().__init__(save_path)

    def _create_index(self, corpus: list[str]) -> IndexFlatIP:
        """
        Create the index

        Args:
            corpus (list[str]): list of all chunks

        Returns:
            Any: return index
        """
        batch_size = 4 if self.mode.get_mode == "cpu" else 256

        model = get_model(mode=self.mode)

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
        """
        Save index

        Args:
            index (Any): the index to save
        """
        write_index(index, str(self.save_path))
