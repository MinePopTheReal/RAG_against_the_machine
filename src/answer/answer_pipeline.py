from ollama import chat
from src.answer.augmenting import Augmenting
from json import loads
from src.utils.file_manager import FileManager
from src.models.models import MinimalAnswer, MinimalSource
from src.message.errors import AnswerError
from httpx import RemoteProtocolError
from tqdm import tqdm

class AnswerPipeline:

    @staticmethod
    def answer_generating_with_query(query:str, retrieved_sources: list[MinimalSource]) -> str:
        model: str = "qwen3:0.6b"

        context = Augmenting(retrieved_sources).create_contexte(query)
        try:
            response = chat(
                model=model,
                messages=context,
                think=False,
                options={
                    "temperature": 0.2,
                    "num_ctx": 6000,
                    "num_predict": 512,
                    "num_batch": 1024,
                    },
            )
        except ConnectionError:
            raise AnswerError(
                "Please start ollama server with this command"
                "in another terminal: 'ollama serve'"
                )
        except RemoteProtocolError:
            raise AnswerError(
                "Please re-start ollama server with this command"
                "in another terminal: 'ollama serve'"
                )
        return response["message"]["content"]

    @staticmethod
    def _load_student_search_results(student_search_results_path: str):
        datas =  FileManager._read(student_search_results_path)

        loaded_datas = loads(datas)
        return loaded_datas

    def answer_generating_with_queries(self, student_search_results_path: str, save_directory: str):
        datas = self._load_student_search_results(student_search_results_path)
        
        responses: list[MinimalAnswer | str | list[dict[str, str] | MinimalSource]] = []

        for data in tqdm(
                datas,
                desc=f"{"Answer":<15.15}",
                colour="cyan",
                unit="query",
                ascii="·■"
            ):
            responses.append(MinimalAnswer(
                question_id=data["question_id"],
                question=data["question"],
                retrieved_sources=data["retrieved_sources"],
                answer=self.answer_generating_with_query(
                    data["question"],
                    [MinimalSource(**source) for source in data["retrieved_sources"]]
                )
            ))

        FileManager.write([response.model_dump(mode="python") for response in responses], save_directory)
