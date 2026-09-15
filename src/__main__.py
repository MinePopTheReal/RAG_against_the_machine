from nncf.common.logging.logger import set_log_level
from datasets import disable_progress_bar
from src.message.errors import Error
from src.cli.cli import CLI
from logging import ERROR
from errno import ENOSPC
from os import environ
from fire import Fire


environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
environ["HF_HUB_VERBOSITY"] = "error"

disable_progress_bar()

set_log_level(ERROR)


def main() -> None:
    """
    main function that launches the CLI
    """
    Fire(CLI)


if __name__ == "__main__":
    try:
        main()
    except Error as e:
        print(e)
    except OSError as e:
        if e.errno == ENOSPC:
            print("No more disk space available")
    except KeyboardInterrupt:
        print(Error("You stop the program"))
