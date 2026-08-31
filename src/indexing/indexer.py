from faiss import write_index, read_index, IndexFlatIP
from sentence_transformers import SentenceTransformer, export_static_quantized_openvino_model
from src.utils.default_path import DefaultPath
from abc import ABC, abstractmethod
from bm25s import tokenize, BM25
from typing import Any
from pathlib import Path


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
        print("created index")
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

    def _save_index(self, index):
        index.save(self.save_path)


class IndexerSemanticEmbedding(Indexer):
    def __init__(
        self,
        corpus: list[str],
        save_path: str = DefaultPath.semantic_index,
        model_name: str = 'BAAI/bge-small-en-v1.5'
    ):
        self.quantized_file_name = "openvino_model_qint8_quantized.xml"
        self.export_path: str = "data/bge-small-openvino"
        super().__init__(corpus, save_path)
        self.model_name = model_name 

    def _get_quantized_model(self) -> SentenceTransformer:
        quantized_file = Path(self.export_path) / self.quantized_file_name
        export_dir = Path(self.export_path)
        quantized_file = export_dir / quantized_file
        config_file = export_dir / "config.json"

        if not (quantized_file.exists() and config_file.exists()):
            print("Export du modèle quantifié (une seule fois)...")
            base_model = SentenceTransformer(self.model_name, backend="openvino")
            # base_model.save_pretrained(self.export_path)  # écrit config.json, tokenizer, etc.
            export_static_quantized_openvino_model(
                base_model,
                quantization_config=None,
                model_name_or_path=self.export_path,
            )
        else:
            print("Modèle quantifié déjà présent, export sauté.")

        return SentenceTransformer(
            Path(self.export_path) ,
            backend="openvino",
            model_kwargs={"file_name": quantized_file},
        )

    def _create_index(self, corpus: list[str]) -> IndexFlatIP:
        import time
        # model = self._get_quantized_model()
        start = time.perf_counter()
        model = SentenceTransformer(self.model_name, backend="openvino")

        export_static_quantized_openvino_model(
            model,
            quantization_config=None,
            model_name_or_path="data/processed",
        )
        print(
            f"Encoding: "
            f"{time.perf_counter() - start:.2f}s"
        )

        start = time.perf_counter()
        vector = model.encode(
            corpus,
            normalize_embeddings=True,
            batch_size=4,
            show_progress_bar=True
        )
        print(
            f"Encoding: "
            f"{time.perf_counter() - start:.2f}s"
        )

        index = IndexFlatIP(vector.shape[1])
        index.add(vector)

        return index

    def _save_index(self, index):
        write_index(index, str(self.save_path))






# if __name__ == "__main__":
#     corpus = ["This is an example sentence", "Each sentence is converted"]

#     Indexer.semantic_embedding(corpus)

# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np

# model = SentenceTransformer("all-MiniLM-L6-v2")  # même modèle pour les deux phases

# # --- Phase 1 : indexation (une fois, hors ligne) ---
# chunks = [...]  # tes chunks de code/doc, les mêmes que ceux donnés à BM25
# vectors = model.encode(chunks, normalize_embeddings=True)  # shape (n_chunks, 384)

# index = faiss.IndexFlatIP(vectors.shape[1])  # IP = inner product, marche bien avec des vecteurs normalisés
# index.add(np.array(vectors))
# faiss.write_index(index, "chunks.index")  # persisté sur disque, pas besoin de refaire ça à chaque run

# # --- Phase 2 : requête (à chaque question) ---
# index = faiss.read_index("chunks.index")
# query_vector = model.encode([question], normalize_embeddings=True)

# distances, indices = index.search(np.array(query_vector), k=5)
# top_chunks = [chunks[i] for i in indices[0]]





# model = SentenceTransformer('BAAI/bge-small-en-v1.5')
# 
# corpus : pas de préfixe
# corpus_embeddings = model.encode(corpus, normalize_embeddings=True)
# 
# question : avec le préfixe recommandé par BAAI
# query_prefix = "Represent this sentence for searching relevant passages: "
# query_embedding = model.encode(query_prefix + question, normalize_embeddings=True)




# from pathlib import Path
# from faiss import IndexFlatIP, read_index, write_index
# from sentence_transformers import SentenceTransformer, export_static_quantized_openvino_model


# class IndexerSemanticEmbedding(Indexer):
#     QUANTIZED_FILE = "openvino_model_qint8_quantized.xml"

#     def __init__(
#         self,
#         corpus: list[str],
#         save_path: str = DefaultPath.semantic_index,
#         model_name: str = 'BAAI/bge-small-en-v1.5',
#         model_export_path: str = "data/bge-small-openvino",
#         batch_size: int = 4,  # reviens à ta valeur benchmarkée
#     ):
#         super().__init__(corpus, save_path)
#         self.model_name = model_name
#         self.model_export_path = model_export_path
#         self.batch_size = batch_size

#         # La partie qui manquait : recharger la version RÉELLEMENT quantifiée


#     def _create_index(self) -> IndexFlatIP:
#         model = self._get_quantized_model()

#         vector = model.encode(
#             self.corpus,
#             normalize_embeddings=True,
#             batch_size=self.batch_size,
#             show_progress_bar=True,
#         )

#         index = IndexFlatIP(vector.shape[1])
#         index.add(vector)
#         return index

#     def _load_index(self):
#         return read_index(self.save_path)

#     def _save_index(self, index):
#         write_index(index, str(self.save_path))