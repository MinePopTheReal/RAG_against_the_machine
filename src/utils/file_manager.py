from src.message.errors import Error
from pydantic import BaseModel
from json import dump, loads
from typing import TypeVar
from pathlib import Path
from typing import Any


class FileManager:
    """
    makes it easier to manage files
    """
    T = TypeVar('T', bound=BaseModel)

    def load(self, file_path: str, return_type: type[T]) -> T:
        """
        allows you to open and directly edit values in a PyDantic model

        Args:
            file_path (str): the file to open
            return_type (type[T]): the type in which we want
            to format our file

        Returns:
            T: Format our file using a PyDantic template
        """
        data = self.read(file_path)
        try:
            load_data = loads(data)
            result = return_type(**load_data)
        except ValueError as e:
            raise Error(
                "The file you tried to open is not a "
                "valid file of the type you specified"
            ) from e
        return result

    @staticmethod
    def read(file_path: str) -> str:
        """
        allows you to read a file

        Args:
            file_path (str): the file path

        Returns:
            str: the contents of the file
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                data = f.read()

            return data
        except IsADirectoryError as e:
            raise Error(
                f"The path points to a folder: {file_path}"
            ) from e

        except FileNotFoundError as e:
            raise Error(
                f"File not found: {file_path}"
            ) from e

        except PermissionError as e:
            raise Error(
                f"Permission denied: {file_path}"
            ) from e

        except OSError as e:
            raise Error(
                f"Unable to read file: {file_path}"
            ) from e

    @staticmethod
    def write(obj: Any, file_path: str) -> None:
        """
        allows you to write to and create a json file
        if it doesn't already exist

        Args:
            obj (Any): A Python object that can be used as JSON
            file_path (str): the file path
        """
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, 'w', encoding="utf-8") as f:
                dump(obj, f, indent=4, ensure_ascii=False)

        except IsADirectoryError as e:
            raise Error(
                f"The path points to a folder: {file_path}"
            ) from e

        except FileNotFoundError as e:
            raise Error(
                f"File not found: {file_path}"
            ) from e

        except PermissionError as e:
            raise Error(
                f"Permission denied: {file_path}"
            ) from e

        except UnicodeEncodeError as e:
            raise Error(
                "Error when encode"
            ) from e

        except TypeError as e:
            raise Error(
                "Object is not JSON serializable"
            ) from e

        except OSError as e:
            raise Error(
                f"Unable to write file: {file_path}"
            ) from e
