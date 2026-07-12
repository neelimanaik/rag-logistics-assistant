import re
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --- Heuristic header detection (regulatory + manuals) ---
def is_section_header(line: str) -> bool:
    line = line.strip()

    if not line:
        return False

    # Examples:
    # "1. INTRODUCTION"
    # "CHAPTER 2 – BILL OF ENTRY"
    # "3.1 Filing the Entry"
    if line.isupper():
        return True

    if line.istitle():
        return True
    
    if re.match(r"^\d+(\.\d+)*\s+", line):
        return True

    if line.startswith(("CHAPTER", "SECTION", "ARTICLE")):
        return True

    return False


def structure_aware_chunk(
    pages: List[Document],
    max_chunk_size: int = 800,
    overlap: int = 100
) -> List[Document]:
    """
    Create section-preserving chunks from PDF pages.
    """

    chunks = []
    current_section = "UNKNOWN"
    current_text = []
    page_start = None

    for page in pages:
        page_text = page.page_content
        page_number = page.metadata.get("page", None)

        for line in page_text.split("\n"):
            if is_section_header(line):
                # flush previous section
                if current_text:
                    chunks.append(
                        Document(
                            page_content="\n".join(current_text),
                            metadata={
                                "section": current_section,
                                "page_start": page_start,
                                "page_end": page_number
                            }
                        )
                    )
                    current_text = []

                current_section = line.strip()
                page_start = page_number

            current_text.append(line)

    # flush last section
    if current_text:
        chunks.append(
            Document(
                page_content="\n".join(current_text),
                metadata={
                    "section": current_section,
                    "page_start": page_start,
                    "page_end": page_number
                }
            )
        )

    # --- Size refinement (only within sections) ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=overlap
    )

    final_chunks = []
    for chunk in chunks:
        if len(chunk.page_content) > max_chunk_size:
            sub_chunks = splitter.split_text(chunk.page_content)
            for sub in sub_chunks:
                final_chunks.append(
                    Document(
                        page_content=sub,
                        metadata=chunk.metadata
                    )
                )
        else:
            final_chunks.append(chunk)

    return final_chunks
