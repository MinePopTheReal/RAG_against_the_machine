from pathlib import Path

class DefaultPath:
    repo = "data/raw/"

    answer = "data/datasets/AnswerQuestions/"
    unanswer = "data/datasets/UnanswerQuestions/"

    output = "data/processed/"

    chunked_source = Path(output) / "chunked_source.json"
    minimal_source = Path(output) / "minimal_source.json"

    bm25_index = Path(output) / "indexs/bm25"
    semantic_index = Path(output) / "indexs/faiss_index.index"





