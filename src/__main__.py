from src.cli.cli import FlagManagement
# from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from src.indexing.indexing_pipeline import IndexingPipeline


def main():
    flag_manager = FlagManagement()
    flags = flag_manager.flags_retrieve()

    # if flags.
    # IndexingPipeline(50).browse_raw_data()

    # import bm25s

    # # Create your corpus here
    # corpus = [
    #     "a cat is a feline and likes to purr",
    #     "a dog is the human's best friend and loves to play",
    #     "a bird is a beautiful animal that can fly",
    # ]

    # # Tokenize the corpus and index it


    # # You can now search the corpus with a query
    # query = "does the fish purr like a cat?"

    # print(docs)
    # print(scores)
    # print(f"Best result (score: {scores[0, 0]:.2f}): {docs[0, 0]}")

    # # Happy with your index? Save it for later...
    # retriever.save("bm25s_index_animals")

    # # ...and load it when needed
    # ret_loaded = bm25s.BM25.load("bm25s_index_animals", load_corpus=True)

if __name__ == "__main__":
    try:
        main()
    except IndentationError as e:
        print(e)
