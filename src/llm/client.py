from dotenv import load_dotenv
load_dotenv()

import os
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

MODEL = os.getenv("AZURE_OPENAI_MODEL")


def generate_answer(context, question):

    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": question}
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.1
    )

    return response.choices[0].message.content

def stream_answer(context, question):

    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": question}
    ]

    stream = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    temperature=0.1,
    stream=True
    )

    full_text = ""

    for chunk in stream:
        if chunk.choices:
            delta = chunk.choices[0].delta.content
            if delta:
                print(delta, end="", flush=True)
                full_text += delta

    print("\n")
    return full_text
