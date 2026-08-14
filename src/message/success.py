from src.message.colors import TerminalColors as TC

class SuccessMessage:
    """
    Represents a formatted success message.

    Adds success formatting when converted to a string.
    """
    def __init__(self, message: str):
        self.message = message

    def __str__(self) -> str:
        return (
            f"{TC.BOLD}{TC.GREEN}[Success]{TC.END}: "
            f"{self.message}"
        )