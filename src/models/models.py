from pydantic import BaseModel, Field, ConfigDict
from uuid import uuid4


class MinimalSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    model_config = ConfigDict(extra="ignore")
    question_id: str = Field(default_factory=lambda: str(uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    model_config = ConfigDict(extra="ignore")
    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    model_config = ConfigDict(extra="ignore")
    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    model_config = ConfigDict(extra="forbid")
    answer: str


class StudentSearchResults(BaseModel):
    model_config = ConfigDict(extra="forbid")
    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    search_results: list[MinimalAnswer]
    k: int
