from src.message.colors import TerminalColors as TC


class Error(Exception):
    """
    The entire logic behind the creation of custom errors.
    """
    default_type = "Not defined"

    def __init__(
        self,
        message: str = "",
        type: str | None = None,
    ) -> None:
        self.message: str = message
        self.type: str = type or self.default_type

        super().__init__(message)

    def __str__(self) -> str:
        """
        Return the formatted error message.
        """
        return (
            f"{TC.BOLD}{TC.RED}[Error]{TC.END}: {self.message}"
            f"{' (' + self.type + ')' if self.type else ''}"
        )

class PydanticError(Error):
    def __init__(
        self, 
        message,
        error, 
        type_error: type[Error] = Error
    ):
        formatted_message = "\n".join(
            f"{message}: {err['msg']}"
            for err in error.errors()
            )

        super().__init__(formatted_message, type_error.default_type)


class CliError(Error):
    """Exception raised by the command line interface."""

    default_type = "cli"


class IndexingError(Error):
    """Exception raised during indexing phase."""

    default_type = "indexing"


class RetrievingError(Error):
    """Exception raised during retrieving phase."""

    default_type = "retrieving"


class AnswerError(Error):
    """Exception raised during answer phase."""

    default_type = "answer"


class EvaluateError(Error):
    """Exception raised during evaluation phase."""

    default_type = "evaluate"
