from src.message.colors import TerminalColors as TC


class WarningMessage:
    """
    Represents a formatted warning message.

    Adds warning formatting when converted to a string.
    """
    def __init__(self, message: str):
        self.message = message

    def __str__(self) -> str:
        return (
            f"{TC.BOLD}{TC.MAGENTA}[Warning]{TC.END}: "
            f"{self.message}"
        )
