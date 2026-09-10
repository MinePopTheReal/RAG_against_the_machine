from nncf.common.logging.logger import set_log_level
from src.message.errors import Error
from src.cli.cli import Flags
from fire import Fire 
import datasets
import logging
import os


os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"

datasets.disable_progress_bar()

set_log_level(logging.ERROR)


def main() -> None:
    Fire(Flags)


if __name__ == "__main__":
    try:
        main()
    except Error as e:
        print(e)
    except KeyboardInterrupt:
        print(Error("You stop the program"))
