import config
from openai import OpenAI

open_ai_client = OpenAI(
    api_key=config.OPENAI_KEY,
    base_url=config.OPENAI_URL,
)

def gpt_request(system_prompt: str,
                client_prompt: str,
                assistant_prompt: str) -> str | None:
    chat_completion = open_ai_client.chat.completions.create(
            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": client_prompt,
                },
                {
                    "role": "assistant",
                    "content": assistant_prompt,
                },
            ],
            model = config.OPENAI_MODEL_NAME, # type: ignore
        )
    return chat_completion.choices[0].message.content
