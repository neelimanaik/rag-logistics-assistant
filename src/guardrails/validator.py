def validate_query(question):

    blocked_topics = ["politics", "religion", "violence"]

    if any(word in question.lower() for word in blocked_topics):
        return False, "This assistant only supports logistics and customs queries."

    return True, None