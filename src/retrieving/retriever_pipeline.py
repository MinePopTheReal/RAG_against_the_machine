from bm25s import BM25, tokenize
from src.message.errors import RetrievingError
from src.utils.file_manager import FileManager
from json import loads
from src.models.models import MinimalSearchResults, MinimalSource
from numpy import array
from src.utils.default_path import DefaultPath
from tqdm import tqdm
from src.indexing.indexer import IndexerBm25, IndexerSemanticEmbedding

class RetrieverPipeline:
    def __init__(self, k: int) -> None:
        self.k = k

    @staticmethod
    def _load_index_json(minimal_source_file_path: str):
        minimal_source = FileManager._read(minimal_source_file_path)

        return loads(minimal_source)

    @staticmethod
    def _load_datasets(datasets_file_path: str) -> list[dict[str, str]]:
        data = FileManager._read(datasets_file_path)

        data_obj = loads(data)
        return data_obj['rag_questions']

    @staticmethod
    def loads_index():
        index_bm25 = IndexerBm25(DefaultPath.bm25_index)._load_index()
        index_embedding = IndexerSemanticEmbedding(DefaultPath.semantic_index)._load_index()

        return (index_bm25, index_embedding)

    def retrieve_chunks_for_query(self, query: str) -> MinimalSource:
        indexs = self.loads_index()

        minimal_source = self._load_index_json(DefaultPath.minimal_source)

        # loaded_minimal_source = array([minimal_source])



        # query_tokens = tokenize(query)

        # docs, scores = retriever.retrieve(query_tokens, k=self.k)

        # docs = [doc for i, doc in enumerate(docs) if scores[0][i]]
        # retieval_minimal_source = []

        # if not docs:
        #     raise RetrievingError("No chunks were found.")

        # for doc in docs[0]:
        #     minimal_source = loaded_minimal_source[0][doc]
        #     retieval_minimal_source.append(MinimalSource(**minimal_source))

        # return retieval_minimal_source

    def retrieve_chunks_for_dataset(self, dataset_path, save_directory):
        search_results: list[MinimalSearchResults] = []

        datas = self._load_datasets(dataset_path)

        for data in tqdm(
            datas,
            desc=f"{"Retriving":<15.15}",
            colour="cyan",
            unit="queries",
            ascii="·■"
        ):
            retieval_minimal_source = self.retrieve_chunks_for_query(data["question"])

            search_results.append(
                MinimalSearchResults(
                    question_id=data['question_id'],
                    question=data['question'],
                    retrieved_sources=retieval_minimal_source
                    )
                )

        save_result = []
        for result in search_results:
            save_result.append(
                {
                    "question_id": result.question_id,
                    "question": result.question,
                    "retrieved_sources": [{
                        "file_path": source.file_path,
                        "first_character_index": source.first_character_index,
                        "last_character_index": source.last_character_index
                        } for source in result.retrieved_sources]
                }
            )

        FileManager.write(save_result, save_directory)

