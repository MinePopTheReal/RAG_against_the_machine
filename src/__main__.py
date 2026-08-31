from src.cli.cli import FlagManagement
# from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from src.indexing.indexing_pipeline import IndexingPipeline
from src.message.errors import Error


def main():
    flag_manager = FlagManagement()
    flag_manager.flags_retrieve()

if __name__ == "__main__":
    try:
        main()
    except Error as e:
        print(e)
    except KeyboardInterrupt:
        print(Error("You stop the program"))
