from pathlib import Path


class DefaultPath:
    """
    # basic default path
    """
    repository = "data/raw/vllm-0.10.1"

    output = "data/processed/"

    chunked_source = str(Path(output) / "chunked_source.json")
    minimal_source = str(Path(output) / "minimal_source.json")

    bm25_index = str(Path(output) / "indexs/bm25")
    semantic_index = str(Path(output) / "indexs/faiss_index.index")

    metadata = str(Path(output) / "metadata.json")
