from openai import AsyncOpenAI
import config

client: AsyncOpenAI = AsyncOpenAI(
    base_url=config.BASE_URL,
    api_key=config.API_KEY,
)

MODEL_CONTEXT_LIMIT: int = 64000