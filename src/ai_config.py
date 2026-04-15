from openai import OpenAI
import config

client: OpenAI = OpenAI(
    base_url=config.BASE_URL,
    api_key=config.API_KEY,
)