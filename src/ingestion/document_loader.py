import os
from pypdf import PdfReader


def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def load_pdf_file(path: str) -> str:
    reader = PdfReader(path)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text() 
        if text:
            pages_text.append(text)
        
    return "\n".join(pages_text)

def load_documents(folder_path:str):
    """
    Docstring for load_documents
    
    :param folder_path: Description
    :type folder_path: str
    Loads all txt, docx and pdf files  from the given folder
    Returns with list of dicts with filename and content
    """
    documents =[]
    allowed_extensions = (".txt",".pdf",".docx")
    for file in os.listdir(folder_path):
        if file.endswith(allowed_extensions):
            full_path = os.path.join(folder_path, file)
            if file.lower().endswith(".txt"):
                text = load_text_file(full_path)
            elif file.lower().endswith(".pdf"):
                text = load_pdf_file(full_path)
            else:
                continue  # Skip unsupported file types for now

        if text.strip():  # Ensure text is not empty
            documents.append({
                "doc_id": file,
                "text": text
            })

    return documents