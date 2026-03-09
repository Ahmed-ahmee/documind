from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

class DocumentLoader:
    """
    Loads documents from a file or directory.
    Supports: PDF, TXT
    Returns: List of LangChain Document objects
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        
    def load(self) -> List[Document]:
        """
        Auto-detects if path is a file or directory
        and loads accordingly.
        """

        if not self.data_path.exists():
            raise FileNotFoundError(f"Path not found: {self.data_path}")
        
        if self.data_path.is_file():
            return self._load_single_file(self.data_path)
        
        if self.data_path.is_dir():
            return self._load_directory(self.data_path)
        
        raise ValueError(f"Path is neither file nor Directory:{self.data_path}")

    def _load_single_file(self, file_path: Path) -> List[Document]:
        ext = file_path.suffix.lower()

        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {ext}. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )

        print(f"  Loading: {file_path.name}")

        if ext == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif ext == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")

        documents = loader.load()

        # ✅ Keep ONLY what we need — clean metadata for citations
        for doc in documents:
            page = doc.metadata.get("page", 0)
            doc.metadata = {
                "source"    : file_path.name,
                "file_type" : ext,
                "page"      : page,
            }

        return documents
    
    def _load_directory(self, dir_path:Path)->List[Document]:
        all_documents = []
        files = [
            f for f in dir_path.iterdir()
            if f.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]


        if not files:
            raise ValueError(f"No supported files found in: {dir_path}")
        
        print(f'Found {len(files)} file(s) in {dir_path}')

        for file_path in files:
            docs = self._load_single_file(file_path)
            all_documents.extend(docs)

        print(f"Total pages/sections loaded: {len(all_documents)}")
        return all_documents
