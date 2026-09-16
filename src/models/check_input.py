from src.utils.file_manager import FileManager
from src.utils.default_path import DefaultPath
from typing import Literal, Self
from torch import cuda
from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    StrictInt,
    field_validator,
    model_validator
    )


class ModeModel(BaseModel):
    """
    A pydantic model for check validy of mode
    """
    mode: Literal["cuda", "cpu", "bm25-only", "default"] = "default"

    @field_validator('mode', mode="after")
    @classmethod
    def set_mode(
        cls,
        value: Literal["cuda", "cpu", "bm25-only", "default"] = "default"
    ) -> str:
        """
        if mode are not usable if current device change it to a valid one

        Args:
            value (Literal['cuda', 'cpu', 'bm25-only', 'default'): the
            different modes. Defaults to "default".

        Returns:
            str: the correct mode
        """
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
        """
        allows us to determine whether the embedding step is feasible
        with current mode

        Returns:
            bool: True if possible, otherwise False
        """
        return False if self.get_mode == "bm25-only" else True

    @property
    def get_mode(self) -> str:
        """
        allows you to enter the mode

        Returns:
            str: the mode
        """
        return self.mode

    def load_mode(self) -> None:
        """
        allows you to enter the mode
        """
        self.mode = FileManager().load(DefaultPath.metadata, ModeModel).mode

    def save_mode(self) -> None:
        """
        allows you to save the mode
        """
        FileManager.write({"mode": self.get_mode}, DefaultPath.metadata)


class CheckIndex(BaseModel):
    """
    allows you to verify the index inputs
    """
    model_config = ConfigDict(extra='forbid')
    max_chunk_size: StrictInt = Field(ge=1)
    repository_path: str = Field(min_length=1)
    chunk_overlap: StrictInt = Field(ge=0)
    mode: ModeModel

    @model_validator(mode="after")
    def check_overlap_and_size(self) -> Self:
        """Check if overlap > chunk size

        Returns:
            Self: Self
        """
        if self.chunk_overlap >= self.max_chunk_size:
            raise ValueError(
                "chunk_overlap must be "
                "smaller than max_chunk_size"
            )
        return self


class CheckSearch(BaseModel):
    """
    allows you to verify the search inputs
    """
    model_config = ConfigDict(extra='forbid')
    query: str = Field(min_length=1)
    k: StrictInt = Field(ge=1)


class CheckSearchDataset(BaseModel):
    """
    allows you to verify the search_dataset inputs
    """
    model_config = ConfigDict(extra='forbid')
    dataset_path: str = Field(min_length=1)
    save_directory: str = Field(min_length=1)
    k: StrictInt = Field(ge=1)


class CheckAnswer(BaseModel):
    """
    allows you to verify the answer inputs
    """
    model_config = ConfigDict(extra='forbid')
    query: str = Field(min_length=1)
    k: StrictInt = Field(ge=1)


class CheckAnswerDataset(BaseModel):
    """
    allows you to verify the answer_dataset inputs
    """
    model_config = ConfigDict(extra='forbid')
    student_search_results_path: str = Field(min_length=1)
    save_directory: str = Field(min_length=1)


class CheckEvaluate(BaseModel):
    """
    allows you to verify the evaluation inputs
    """
    model_config = ConfigDict(extra='forbid')
    student_search_results_path: str = Field(min_length=1)
    dataset_path: str = Field(min_length=1)
