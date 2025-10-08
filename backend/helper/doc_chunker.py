from backend import config
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter


class Chunker:
    def __init__(self, path_folder: str = config.doc_path):
        """
        Constructor for instantiating class Chunker
        :param path_folder: where folders a
        :return: None
        """
        self.path_folder = path_folder

    def load_documents(self) -> list | str | None:
        """
        Reads in documents using LangChain's loaders
        Goes through everything in the sub-folders of the generated documents
        :return: list (read in documents)
        """

        base_path = Path(self.path_folder)
        if base_path.exists():
            # Get all subfolders (directories only)
            folders = [f for f in base_path.iterdir() if f.is_dir()]
            loader_kwargs = {'encoding': 'utf-8'}

            documents = []
            if folders:
                for folder in folders:
                    doc_type = folder.name
                    loader = DirectoryLoader(
                        str(folder),
                        glob="**/*.md",
                        loader_cls=TextLoader,
                        loader_kwargs=loader_kwargs
                    )
                    folder_docs = loader.load()
                    for doc in folder_docs:
                        doc.metadata["doc_type"] = doc_type
                        documents.append(doc)
            else:
                return f"No subfolders found in {base_path.resolve()}"
        else:
            return f"The specified path does not exist: {base_path.resolve()}"
        return documents

    def chunk(self) -> list:
        """
        chunk documents
        :return: list (list of chunked docs)
        """
        loaded_docs = self.load_documents()
        if isinstance(loaded_docs, list):
            text_splitter = CharacterTextSplitter(chunk_size=860, chunk_overlap=150)
            chunks = text_splitter.split_documents(loaded_docs)
            return chunks
        return []


if __name__ == "__main__":
    chunker = Chunker()
    docs = chunker.chunk()
    r_chunks = chunker.chunk()
    print(len(r_chunks))
    print(f"Document types: {', '.join((set(chunk.metadata['doc_type'] for chunk in r_chunks)))}")
