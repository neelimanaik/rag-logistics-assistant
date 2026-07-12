import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Load .env ONCE here; every module imports the `settings` object below instead
# of calling os.getenv() on its own. Single source of truth.
load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Single source of truth for configuration.

    LLM_PROVIDER selects the backend: "ollama" (local, default), "openai", or
    "azure". Everything else reads from one place, so the name-mismatch and
    version-drift bugs we fixed earlier cannot recur.
    """

    # Which backend to use.
    llm_provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "ollama").lower()
    )

    # ---- Model names (accept generic names, fall back to the old Azure ones) ----
    chat_model: str = field(
        default_factory=lambda: os.getenv("CHAT_MODEL")
        or os.getenv("AZURE_OPENAI_MODEL")
        or "llama3.2"
    )
    embed_model: str = field(
        default_factory=lambda: os.getenv("EMBED_MODEL")
        or os.getenv("AZURE_EMBED_MODEL")
        or "nomic-embed-text"
    )

    # ---- Ollama / OpenAI-compatible endpoint ----
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    )
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )

    # ---- Azure OpenAI (only used when LLM_PROVIDER=azure) ----
    azure_openai_api_key: str = field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_API_KEY", "")
    )
    azure_openai_endpoint: str = field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT", "")
    )
    azure_openai_api_version: str = field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    )

    # ---- Paths ----
    index_path: str = field(
        default_factory=lambda: os.getenv("INDEX_PATH", "data/processed/index")
    )

    def validate(self):
        """Fail fast with a clear message if required settings are missing.
        Requirements depend on the provider. Call at startup, not at import."""
        missing = []
        if self.llm_provider == "azure":
            if not self.azure_openai_api_key:
                missing.append("AZURE_OPENAI_API_KEY")
            if not self.azure_openai_endpoint:
                missing.append("AZURE_OPENAI_ENDPOINT")
        elif self.llm_provider == "openai":
            if not self.openai_api_key:
                missing.append("OPENAI_API_KEY")
        # "ollama" needs no key (it runs locally).
        if not self.chat_model:
            missing.append("CHAT_MODEL")
        if not self.embed_model:
            missing.append("EMBED_MODEL")
        if missing:
            raise RuntimeError(
                "Missing required configuration: "
                + ", ".join(missing)
                + ". Set these in your .env file."
            )
        return self


settings = Settings()
