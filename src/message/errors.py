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
        tutorial_message: str = ""
    ) -> None:
        self.message: str = message
        self.type: str = type or self.default_type
        self.tutorial: str = tutorial_message

        super().__init__(message)

    def __str__(self) -> str:
        """
        Return the formatted error message.
        """
        return (
            f"{TC.BOLD}{TC.RED}[Error]{TC.END}: {self.message}"
            f"{' (' + self.type + ')' if self.type else ''}"
            f"\n {self.tutorial}"
        )


class IndexingError(Error):
    """Exception raised during indexing phase."""

    default_type = "indexing"

class RetrievingError(Error):
    """Exception raised during retrieving phase."""

    default_type = "retrieving"

class AnswerError(Error):
    """Exception raised during answer phase."""

    default_type = "answer"

class CliError(Error):
    """Exception raised by the command line interface."""

    default_type = "cli"


class MesssageError:
    """
    Stores error message templates and syntax information.

    Provides predefined syntax descriptions and metadata hints used
    to display helpful error messages.
    """
    pass
