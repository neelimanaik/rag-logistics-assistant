from openai import OpenAI, AzureOpenAI

from src.config.settings import settings


def get_client():
    """Return an OpenAI-compatible client for the configured provider.

    Ollama and OpenAI both speak the OpenAI API, so they share the OpenAI class;
    only Azure needs its own class. This single seam is what lets us switch the
    backend with one env var (LLM_PROVIDER) and zero code changes anywhere else.
    """
    provider = settings.llm_provider

    if provider == "azure":
        return AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
        )

    if provider == "openai":
        return OpenAI(api_key=settings.openai_api_key)

    # default: ollama (local). Ollama ignores the api_key, but the SDK requires
    # some non-empty value, so we pass a dummy.
    return OpenAI(api_key="ollama", base_url=settings.ollama_base_url)
