from src.message.errors import Error
from json import dump
from typing import Any
from pathlib import Path

class FileManager:
    @staticmethod
    def _read(file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                data = f.read()
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

        return data

    @staticmethod
    def write(obj: list[Any], file_path: str):
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
                f"Error when encode"
            ) from e

        except TypeError as e:
            raise Error(
                f"Object is not JSON serializable"
            ) from e

        except OSError as e:
            raise Error(
                f"Unable to write file: {file_path}"
            ) from e

