from langchain_community.document_loaders import PyPDFLoader
from typing import List
from langchain_core.documents import Document


def load_pdf(path: str) -> List[Document]:
    """
    Load a PDF and return LangChain Document objects,
    one per page, with page metadata.
    """
    loader = PyPDFLoader(path)
    documents = loader.load()

    return documents
