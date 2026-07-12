from src.config.settings import settings
from src.llm.provider import get_client

# One client, chosen by LLM_PROVIDER (ollama / openai / azure).
client = get_client()
MODEL = settings.chat_model


def generate_answer(context, question):
    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": question},
    ]
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.1,
    )
    return response.choices[0].message.content


def stream_answer_tokens(context, question):
    """Yield answer tokens (deltas) as they arrive. This function only PRODUCES
    tokens; the caller decides where they go. That separation is what lets the
    API layer stream them to an HTTP client."""
    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": question},
    ]
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.1,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


def stream_answer(context, question):
    """Console helper: prints tokens and returns the full text. Built on the
    generator so there is one place that talks to the LLM."""
    full_text = ""
    for delta in stream_answer_tokens(context, question):
        print(delta, end="", flush=True)
        full_text += delta
    print()
    return full_text
