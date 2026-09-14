from pydantic import BaseModel, Field, ConfigDict, StrictInt, field_validator
from typing import Literal
from torch import cuda
from src.utils.file_manager import FileManager
from src.utils.default_path import DefaultPath


class ModeModel(BaseModel):
    mode: Literal["cuda", "cpu", "bm25-only", "default"] = "default"

    @field_validator('mode', mode="after")
    @classmethod
    def set_mode(
        cls,
        value: Literal["cuda", "cpu", "bm25-only", "default"] = "default"
    ) -> str:
        mode = value

        if value != "default":
            if value == "cuda" and cuda.is_available():
                mode = "cuda"
            elif value != "bm25-only":
                mode = "cpu"
        else:
            mode = "cuda" if cuda.is_available() else "cpu"

        return mode

    @property
    def can_embed(self) -> bool:
        return False if self.get_mode == "bm25-only" else True

    @property
    def get_mode(self) -> str:
        return self.mode

    def load_mode(self) -> None:
        self.mode = FileManager().load(DefaultPath.metadata, ModeModel).mode

    def save_mode(self) -> None:
        FileManager.write({"mode": self.get_mode}, DefaultPath.metadata)


class CheckIndex(BaseModel):
    model_config = ConfigDict(extra='forbid')
    max_chunk_size: StrictInt = Field(ge=1)
    repository_path: str = Field(min_length=1)
    chunk_overlap: StrictInt = Field(ge=0)
    mode: ModeModel


class CheckSearch(BaseModel):
    model_config = ConfigDict(extra='forbid')
    query: str = Field(min_length=1)
    k: StrictInt = Field(ge=1)


class CheckSearchDataset(BaseModel):
    model_config = ConfigDict(extra='forbid')
    dataset_path: str = Field(min_length=1)
    save_directory: str = Field(min_length=1)
    k: StrictInt = Field(ge=1)


class CheckAnswer(BaseModel):
    model_config = ConfigDict(extra='forbid')
    query: str = Field(min_length=1)
    k: StrictInt = Field(ge=1)


class CheckAnswerDataset(BaseModel):
    model_config = ConfigDict(extra='forbid')
    student_search_results_path: str = Field(min_length=1)
    save_directory: str = Field(min_length=1)


class CheckEvaluate(BaseModel):
    model_config = ConfigDict(extra='forbid')
    student_search_results_path: str = Field(min_length=1)
    dataset_path: str = Field(min_length=1)
