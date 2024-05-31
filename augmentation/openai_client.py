import config
from openai import OpenAI

open_ai_client = OpenAI(
    api_key=config.OPENAI_KEY,
    base_url=config.OPENAI_URL,
)

def add_shot(messages: list[dict[str,str]],
             client_message: str,
             assistant_message: str | None):
    messages.append(
        {
            "role": "user",
            "content": client_message,
        }
    )
    if assistant_message:
        messages.append(
            {
                "role": "assistant",
                "content": assistant_message,
            }
        )

def gpt_request(system_prompt: str,
                client_prompt: str,
                assistant_prompt: str | None = None,
                few_shot: list[tuple[str, str]] | None = None) -> str | None:

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
    ]

    if few_shot:
        for client_message, assistant_message in few_shot:
            add_shot(messages, client_message, assistant_message)

    add_shot(messages, client_prompt, assistant_prompt)

    chat_completion = open_ai_client.chat.completions.create(
            messages = messages, # type: ignore
            model = config.OPENAI_MODEL_NAME, # type: ignore
        )
    return chat_completion.choices[0].message.content
