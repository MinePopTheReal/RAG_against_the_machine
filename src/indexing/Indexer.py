from bm25s import tokenize, BM25
from src.indexing.chunker import MinimalSource
from tqdm import tqdm

class Indexer:
    def __init__(self):
        pass

    def bm25(self, corpus: list[str]) -> BM25:

        corpus_tokens = [
            tokenize(chunk)
            for chunk in tqdm(
                corpus,
                desc=f"{"Tokenizing":<15.15}",
                colour="cyan",
                unit="chunk",
                ascii="·■"
            )
        ]
        corpus_tokens = tokenize(corpus)
        retriever = BM25(corpus=corpus)
        retriever.index(corpus_tokens)

        return retriever