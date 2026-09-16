from pydantic import BaseModel, Field, ConfigDict
from uuid import uuid4


class MinimalSource(BaseModel):
    """
    a representation of a chunk, including its
    path and position in the file
    """
    model_config = ConfigDict(extra="forbid")
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """
    The question and its ID
    """
    model_config = ConfigDict(extra="ignore")
    question_id: str = Field(default_factory=lambda: str(uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """
    The question and its ID, with sources to answer this question and response
    """
    model_config = ConfigDict(extra="ignore")
    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """
    a list of questions, whether answered or not
    """
    model_config = ConfigDict(extra="ignore")
    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """
    The question and its ID, with sources to answer this question
    """
    model_config = ConfigDict(extra="forbid")
    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """
    The question and its ID, with sources to answer this question and response
    """
    model_config = ConfigDict(extra="forbid")
    answer: str


class StudentSearchResults(BaseModel):
    """
    A complete response list including the question,
    question, chunks, generated response, and the k chunk
    """
    model_config = ConfigDict(extra="forbid")
    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """
    a list of answers
    """
    model_config = ConfigDict(extra="forbid")
    search_results: list[MinimalAnswer]
    k: int
