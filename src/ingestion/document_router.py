def infer_document_metadata(file_path: str) -> dict:
    path = file_path.lower()

    if "user_manual" in path:
        return {
            "document_type": "user_manual",
            "application": "eBrokerage",
            "authority": "CBP",
            "confidence_level": "medium"
        }

    if "customs" in path:
        return {
            "document_type": "customs_regulation",
            "application": "eBrokerage",
            "authority": "CBP",
            "confidence_level": "high"
        }

    return {
        "document_type": "unknown"
    }
