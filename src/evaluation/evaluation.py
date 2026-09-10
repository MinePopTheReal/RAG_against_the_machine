from src.message.colors import TerminalColors as TC
from src.utils.file_manager import FileManager
from src.message.errors import EvaluateError
from src.models.models import (
    RagDataset,
    AnsweredQuestion,
    MinimalSource,
    StudentSearchResults,
)
from typing import Any


class Evaluation:
    def __init__(
        self,
        student_search_results_path: str,
        dataset_path: str
    ) -> None:
        self.student_search_results_path = student_search_results_path
        self.dataset_path = dataset_path

    def _load_jsons(self) -> Any:
        student_search_result = FileManager().load(
            self.student_search_results_path,
            StudentSearchResults
        )
        dataset = FileManager().load(self.dataset_path, RagDataset)

        return student_search_result, dataset

    @staticmethod
    def iou_calcul(
        source: MinimalSource,
        dataset_source: MinimalSource
    ) -> float:
        first_index = max(
            source.first_character_index,
            dataset_source.first_character_index
            )
        last_index = min(
            source.last_character_index,
            dataset_source.last_character_index
            )

        inter = last_index - first_index
        if inter < 0:
            return -1

        first_index = min(
            source.first_character_index,
            dataset_source.first_character_index
            )
        last_index = max(
            source.last_character_index,
            dataset_source.last_character_index
            )

        union = last_index - first_index
        if union <= 0:
            return -1
        return inter / union

    def calculation(
        self,
        student_search_result: StudentSearchResults,
        dataset: RagDataset
    ) -> list[float]:
        search_results = student_search_result.search_results
        dataset_id_and_sources = {}
        for data in dataset.rag_questions:
            if not isinstance(data, AnsweredQuestion):
                raise EvaluateError(
                    "You must give a dataset with AnsweredQuestion only"
                )
            dataset_id_and_sources[data.question_id] = data.sources

        recalls = [1, 3, 5, 10]
        results_for_each_recall = []

        for recall in recalls:
            counter = 0

            for search_result in search_results:
                source_for_recall = search_result.retrieved_sources[:recall]

                find = False
                for source in source_for_recall:
                    dataset_sources = dataset_id_and_sources[
                        search_result.question_id
                    ]
                    for dataset_source in dataset_sources:

                        if source.file_path == dataset_source.file_path:
                            if self.iou_calcul(source, dataset_source) >= 0.05:
                                counter += 1
                                find = True
                                break
                    if find:
                        break

            results_for_each_recall.append(counter / len(search_results))
        return results_for_each_recall

    def display_evaluation(self) -> None:
        student_search_result, dataset = self._load_jsons()

        result = self.calculation(student_search_result, dataset)

        print(
            "Evaluation Results\n"
            "========================================\n"
            f"Recall@1: {TC.BOLD}{TC.BLUE}{result[0]:.3f}{TC.END} | "
            f"Recall@3: {TC.BOLD}{TC.BLUE}{result[1]:.3f}{TC.END} | "
            f"Recall@5: {TC.BOLD}{TC.BLUE}{result[2]:.3f}{TC.END} | "
            f"Recall@10: {TC.BOLD}{TC.BLUE}{result[3]:.3f}{TC.END}"
        )
