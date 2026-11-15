"""Factory for creating a LangChain-native LLM specifically for tool calling."""
from langchain_openai import ChatOpenAI
from app.core.config import Settings

def create_tool_calling_llm(settings: Settings) -> ChatOpenAI:
    """
    Create a LangChain ChatOpenAI instance for tool calling.
    Uses the same .env variables as your existing llm_client.py.
    """
    # Gemini requires OpenAI-compatible endpoint for LangChain
    if settings.llm_provider == "gemini":
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    else:
        base_url = settings.llm_base_url or "https://api.openai.com/v1"
    
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=base_url,
        temperature=0.0,
    )